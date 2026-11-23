#!/usr/bin/env python3
"""
Using a fine-tuned model on an ordinary computer, without a GPU (CPU inference).
For every headline: probabilities down / flat / up for BTC, ETH and gold. The forecast of a day uses the mean
over that day's headlines, with both rules of the evaluation: raw "up if P(up) > P(down)" and centered
"up if P(up) - P(down) > threshold" (threshold = val median, read from the run's metrics.json).

  python ft_predict.py --run experiments/2026-11-23_news-direction/modernbert-large/A_next \
      --text "Bitcoin ETF sees record inflows" --text "Fed signals a rate cut"
  python ft_predict.py --run <run dir> --csv news_bd2.csv --date-column date --since 2026-11-15 --out forecast.csv

Needs Python 3.10+ and: pip install torch transformers peft pandas
(the CPU build of torch is enough: pip install torch --index-url https://download.pytorch.org/whl/cpu).
<run dir> is a folder from ft_train.py with config.json and checkpoint/. A full fine-tune (ModernBERT, FinBERT) is
self-contained; for a LoRA run only the small adapter is in checkpoint/, the base model (model id in
config.json) is downloaded from Hugging Face on the first start.
Variant B runs expect the headline + annotation text (text_b in ft_data.py); use variant A for plain headlines.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

ASSETS = ["btc", "eth", "gold"]
CLASSES = ["down", "flat", "up"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run folder with config.json and checkpoint/")
    ap.add_argument("--text", action="append", default=[], help="a headline (repeat for several)")
    ap.add_argument("--csv", help="CSV with headlines (separator is detected)")
    ap.add_argument("--column", default="news", help="column with headlines in --csv")
    ap.add_argument("--date-column", help="column with the news day in --csv -> one forecast per day")
    ap.add_argument("--since", help="with --date-column: only rows from this day on (YYYY-MM-DD)")
    ap.add_argument("--limit", type=int, help="take only the first N rows of --csv")
    ap.add_argument("--out", help="save per-headline probabilities to this CSV")
    ap.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    ap.add_argument("--dtype", choices=["auto", "float32", "bfloat16"], default="auto",
                    help="auto: float32 for models under 2B parameters, else bfloat16 (half the RAM)")
    ap.add_argument("--threads", type=int, help="CPU threads (default: all cores)")
    ap.add_argument("--batch", type=int, default=16)
    return ap.parse_args()


def rss_gb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except ImportError:
        return float("nan")


def load_model(run, device, dtype_arg):
    cfg = json.load(open(os.path.join(run, "config.json"), encoding="utf-8"))
    ckpt = os.path.join(run, "checkpoint")
    head = cfg.get("head", "seq_cls")
    n_out = 3 * len(ASSETS)

    def pick(n_params):
        if dtype_arg != "auto":
            return getattr(torch, dtype_arg)
        return torch.float32 if n_params < 2e9 else torch.bfloat16

    tok = AutoTokenizer.from_pretrained(ckpt)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    if cfg["method"] == "full":
        model = AutoModelForSequenceClassification.from_pretrained(ckpt, dtype=torch.float32)
        model = model.to(pick(model.num_parameters()))
    else:
        from peft import PeftModel
        probe = 4e9 if dtype_arg == "auto" else 0      # LLM backbones are > 2B: load directly in bf16
        dtype = pick(probe)
        if head == "seq_cls":
            base = AutoModelForSequenceClassification.from_pretrained(cfg["model"], num_labels=n_out, dtype=dtype)
            base.config.pad_token_id = tok.pad_token_id
            base.config.get_text_config().pad_token_id = tok.pad_token_id
            model = PeftModel.from_pretrained(base, ckpt).merge_and_unload()
        else:
            from ft_train import LastTokenHead
            lm = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=dtype)
            hidden = lm.config.get_text_config().hidden_size
            decoder = PeftModel.from_pretrained(lm.get_decoder(), ckpt).merge_and_unload()
            del lm
            model = LastTokenHead(decoder, hidden, n_out)
            model.score.load_state_dict(torch.load(os.path.join(ckpt, "score.pt"), map_location="cpu"))
            model.to(dtype)
    model.to(device).eval()
    return model, tok, cfg


@torch.no_grad()
def predict(model, tok, texts, max_len, batch, device):
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), len(ASSETS), 3), dtype=np.float32)
    for s in range(0, len(texts), batch):
        idx = order[s:s + batch]
        enc = tok([texts[i] for i in idx], truncation=True, max_length=max_len, padding=True,
                  return_tensors="pt").to(device)
        logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
        out[idx] = torch.softmax(logits.float().view(len(idx), len(ASSETS), 3), -1).cpu().numpy()
    return out


def centered_thresholds(run, target):
    """Val thresholds of the centered rule saved by the evaluation ({} if the run has no metrics.json)."""
    path = os.path.join(run, "metrics.json")
    if not os.path.exists(path):
        return {}
    day_level = json.load(open(path, encoding="utf-8"))["test"][target]["day_level"]
    return {a: v["threshold_centered"] for a, v in day_level.items() if v.get("threshold_centered") is not None}


def day_forecast(p, thresholds):
    """p: (n, assets, 3) for the headlines of one day ->
    {asset: (signal = P(up) - P(down), raw direction, centered direction or None)}"""
    m = p.mean(0)
    out = {}
    for k, a in enumerate(ASSETS):
        s = float(m[k, 2] - m[k, 0])
        thr = thresholds.get(a)
        out[a] = (s, "up" if s > 0 else "down", None if thr is None else ("up" if s > thr else "down"))
    return out


def main():
    sys.stdout.reconfigure(errors="replace")       # Windows console code pages can't print every headline
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    texts, dates = list(args.text), None
    if args.csv:
        df = pd.read_csv(args.csv, sep=None, engine="python", encoding="utf-8-sig")
        df = df.dropna(subset=[args.column])
        if args.since:
            if not args.date_column:
                raise SystemExit("--since needs --date-column")
            df = df[df[args.date_column].astype(str).str[:10] >= args.since]
        if args.limit:
            df = df.head(args.limit)
        texts += df[args.column].astype(str).tolist()
        if args.date_column:
            dates = ["(--text)"] * len(args.text) + df[args.date_column].astype(str).str[:10].tolist()
    if not texts:
        raise SystemExit("no headlines: use --text or --csv")

    t0 = time.time()
    model, tok, cfg = load_model(args.run, args.device, args.dtype)
    n_params = sum(p.numel() for p in model.parameters())
    dtype = next(model.parameters()).dtype
    load_s = time.time() - t0
    print(f"model {cfg['model']} ({cfg['method']}, variant {cfg['variant']}, target '{cfg['target']}'), "
          f"{n_params / 1e9:.2f}B params, {str(dtype).replace('torch.', '')} on {args.device}, "
          f"{torch.get_num_threads()} threads; loaded in {load_s:.0f} s, RAM {rss_gb():.1f} GB", flush=True)
    if cfg["variant"] == "B":
        print("note: variant B was trained on headline + annotation text, plain headlines are out of format")

    max_len = cfg.get("hyperparameters", {}).get("max_len", 128)
    t1 = time.time()
    p = predict(model, tok, texts, max_len, args.batch, args.device)
    dt = time.time() - t1
    print(f"{len(texts)} headlines in {dt:.1f} s ({len(texts) / dt:.1f} per second), peak RAM {rss_gb():.1f} GB\n")

    rows = []
    for i, t in enumerate(texts):
        row = {"news": t} if dates is None else {"date": dates[i], "news": t}
        for k, a in enumerate(ASSETS):
            for c, name in enumerate(CLASSES):
                row[f"p_{a}_{name}"] = round(float(p[i, k, c]), 4)
        rows.append(row)
    res = pd.DataFrame(rows)
    if args.out:
        res.to_csv(args.out, sep=";", index=False, encoding="utf-8-sig")
        print(f"saved {args.out}")

    for i, t in enumerate(texts[:20]):
        cells = "  ".join(f"{a.upper()} " + "/".join(f"{x:.2f}" for x in p[i, k]) for k, a in enumerate(ASSETS))
        print(f"{t[:70]:<70}  {cells}   (down/flat/up)")
    if len(texts) > 20:
        print(f"... {len(texts) - 20} more")

    groups = {"all headlines": np.arange(len(texts))} if dates is None else \
        {d: np.where(np.array(dates) == d)[0] for d in sorted(set(dates))}
    thresholds = centered_thresholds(args.run, cfg["target"])
    print("\nforecast of the next 24 h by the mean over headlines" if cfg["target"] == "next" else
          "\nforecast (same-day target) by the mean over headlines")
    print("  signal = P(up) - P(down); raw: up if signal > 0" +
          ("; centered: up if signal > val threshold (" +
           ", ".join(f"{a.upper()} {t:+.3f}" for a, t in thresholds.items()) + ")" if thresholds else ""))
    for d, idx in groups.items():
        fc = day_forecast(p[idx], thresholds)
        print(f"  {d} ({len(idx)} headlines): " + ", ".join(
            f"{a.upper()} {s:+.3f} raw {raw}" + (f" / centered {cen}" if cen else "")
            for a, (s, raw, cen) in fc.items()))


if __name__ == "__main__":
    main()
