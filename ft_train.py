#!/usr/bin/env python3
"""
Fine-tune of a language model with a classification head (FINETUNE_PLAN.md, sections 5-7):
one headline -> down / flat / up for BTC, ETH and gold at once (3 heads of 3 classes = 9 logits).
LoRA for decoder LLMs (Qwen, Gemma), full fine-tune for small encoders (ModernBERT, FinBERT); bf16 autocast.
Early stopping on val loss, the best step is kept; then val and test are predicted once.

Run on DGX Spark (inside the container, from the repo root):
  bash experiments/spark_run.sh ft_train.py --model Qwen/Qwen3.5-4B-Base --variant A --target next
  bash experiments/spark_run.sh ft_train.py --model answerdotai/ModernBERT-large --method full --lr 3e-5 ...
  ... --max-steps 300 --tag pilot      # short pilot run
Output: experiments/<exp>/<model>/<variant>_<target>[_<tag>]/ with config.json, train_log.csv, metrics.json,
predictions_*.csv, daily_test.csv, plots/; weights go to checkpoint/ (ignored by git).
"""

import argparse
import math
import os
import random
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from ft_common import (ASSETS, CLASSES, EXP_DIR, TARGETS, TEST_START, THRESHOLDS, VAL_START, evaluate, load_dataset,
                       pcols, report_run, write_json, ycol, _plt)

# public release date = upper bound of the knowledge cutoff; must be before TEST_START (no peeking at test)
RELEASED = {"Qwen/Qwen3.5-4B-Base": "2026-02-27", "Qwen/Qwen3.5-9B-Base": "2026-02-26",
            "answerdotai/ModernBERT-large": "2024-12-11", "ProsusAI/finbert": "2019-08-27",
            "google/gemma-4-E4B": "2026-03-02", "google/gemma-4-12B": "2026-05-23"}
N_ASSETS = len(ASSETS)
HEAD_WORDS = ("score", "classifier", "head")


class LastTokenHead(torch.nn.Module):
    """Classification head on the last real token, for decoders without *ForSequenceClassification (Gemma 4)."""

    def __init__(self, decoder, hidden, n_out):
        super().__init__()
        self.decoder = decoder
        self.score = torch.nn.Linear(hidden, n_out, bias=False)

    def forward(self, input_ids, attention_mask):
        h = self.decoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = h[torch.arange(h.size(0), device=h.device), attention_mask.sum(1) - 1]
        return SimpleNamespace(logits=self.score(pooled.to(self.score.weight.dtype)))


def load_classifier(model_id, n_out, dtype):
    """-> (model, head kind). Standard *ForSequenceClassification if transformers has it, else LastTokenHead."""
    try:
        # ignore_mismatched_sizes: checkpoints with their own classifier (FinBERT: 3 sentiment classes) get a new head
        model, info = AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=n_out, dtype=dtype, output_loading_info=True, ignore_mismatched_sizes=True)
        kind = "seq_cls"
    except ValueError:
        lm, info = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, output_loading_info=True)
        decoder = lm.get_decoder()
        model = LastTokenHead(decoder, lm.config.get_text_config().hidden_size, n_out)
        kind = "last_token"
        del lm
    missing = [k for k in info["missing_keys"] if not any(w in k for w in HEAD_WORDS)]
    replaced = [k[0] if isinstance(k, tuple) else k for k in info.get("mismatched_keys", [])]
    print(f"loaded {type(model).__name__} ({kind}); new head weights: "
          f"{[k for k in info['missing_keys'] if k not in missing] + replaced}; "
          f"unexpected keys: {len(info['unexpected_keys'])}", flush=True)
    if missing:
        raise SystemExit(f"backbone weights were not loaded ({len(missing)}), e.g. {missing[:5]}")
    return model, kind


def add_lora(model, kind, args):
    from peft import LoraConfig, get_peft_model
    if kind == "seq_cls":
        cfg = LoraConfig(task_type="SEQ_CLS", r=args.lora_r, lora_alpha=args.lora_alpha,
                         lora_dropout=args.lora_dropout, target_modules="all-linear")
        model = get_peft_model(model, cfg)
    else:
        cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                         target_modules="all-linear")
        model.decoder = get_peft_model(model.decoder, cfg)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_tr:,} of {n_all:,} ({n_tr / n_all:.2%})", flush=True)
    return model


def save_checkpoint(model, kind, tok, ckpt):
    """Full fine-tune: standalone HF model; LoRA: adapter (+ head for LastTokenHead). Loaded by ft_predict.py."""
    os.makedirs(ckpt, exist_ok=True)
    if kind == "seq_cls":
        model.save_pretrained(ckpt)
    else:
        model.decoder.save_pretrained(ckpt)
        torch.save(model.score.state_dict(), os.path.join(ckpt, "score.pt"))
    tok.save_pretrained(ckpt)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", choices=["A", "B"], default="A", help="A: headline, B: headline + annotation")
    ap.add_argument("--target", choices=TARGETS, default="next")
    ap.add_argument("--method", choices=["lora", "full"], default="lora")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.03, help="share of steps")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--patience", type=int, default=6, help="evaluations without a better val loss")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--eval-batch", type=int, default=128)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    return ap.parse_args()


def short_name(model):
    return model.split("/")[-1].lower()


class Batches:
    """Dynamic padding of pre-tokenized headlines."""

    def __init__(self, ids, labels, pad_id):
        self.ids, self.labels, self.pad_id = ids, labels, pad_id

    def make(self, idx):
        seqs = [self.ids[i] for i in idx]
        n = max(len(s) for s in seqs)
        x = torch.full((len(seqs), n), self.pad_id, dtype=torch.long)
        m = torch.zeros((len(seqs), n), dtype=torch.long)
        for j, s in enumerate(seqs):
            x[j, :len(s)] = torch.tensor(s)
            m[j, :len(s)] = 1
        y = torch.from_numpy(self.labels[idx]) if self.labels is not None else None
        return x, m, y


def labels_of(df, target):
    y = df[[ycol(a, target) for a in ASSETS]].to_numpy(dtype=float)
    return np.where(np.isnan(y), -100, y).astype(np.int64)


def loss_fn(logits, y):
    return F.cross_entropy(logits.float().view(-1, 3), y.view(-1), ignore_index=-100)


@torch.no_grad()
def predict(model, data, idx_all, bs, device):
    model.eval()
    order = sorted(idx_all, key=lambda i: len(data.ids[i]))      # less padding
    probs, losses, counts = {}, 0.0, 0
    for s in range(0, len(order), bs):
        idx = np.array(order[s:s + bs])
        x, m, y = data.make(idx)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=x.to(device), attention_mask=m.to(device)).logits
        logits = logits.float().view(len(idx), N_ASSETS, 3)
        if y is not None:
            yy = y.to(device)
            n = int((yy != -100).sum())
            if n:
                losses += float(loss_fn(logits, yy)) * n
                counts += n
        p = torch.softmax(logits, -1).cpu().numpy()
        for j, i in enumerate(idx):
            probs[i] = p[j]
    model.train()
    p = np.stack([probs[i] for i in idx_all])
    return p, (losses / counts if counts else float("nan"))


def to_frame(ids, p):
    df = pd.DataFrame({"id": ids})
    for k, a in enumerate(ASSETS):
        df[pcols(a)] = p[:, k, :]
    return df


def plot_curves(log, out, title):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    tr = log.dropna(subset=["train_loss"])
    ev = log.dropna(subset=["val_loss"])
    ax.plot(tr["step"], tr["train_loss"], label="train loss", color="#457b9d", alpha=0.8)
    ax.plot(ev["step"], ev["val_loss"], label="val loss", color="#e76f51", marker="o")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy (mean over 3 assets)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "plots", "train_loss.png"), dpi=110)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, ["acc", "f1", "dayacc"]):
        for a in ASSETS:
            ax.plot(ev["step"], ev[f"val_{metric}_{a}"], marker=".", label=a.upper())
        ax.set_title({"acc": "val accuracy (news)", "f1": "val macro-F1 (news)",
                      "dayacc": "val up/down accuracy (days)"}[metric])
        ax.set_xlabel("step")
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "plots", "val_quality.png"), dpi=110)
    plt.close(fig)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"
    run = f"{args.variant}_{args.target}" + (f"_{args.tag}" if args.tag else "")
    out = os.path.join(EXP_DIR, short_name(args.model), run)
    os.makedirs(os.path.join(out, "plots"), exist_ok=True)
    print(f"run -> {out}", flush=True)

    news, days = load_dataset()
    text = "text_a" if args.variant == "A" else "text_b"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    enc = tok(news[text].tolist(), truncation=True, max_length=args.max_len, add_special_tokens=True)["input_ids"]
    data = Batches(enc, labels_of(news, args.target), tok.pad_token_id)
    split = news["split"].to_numpy()
    has_label = (data.labels != -100).any(1)
    tr_idx = np.where((split == "train") & has_label)[0]
    va_idx = np.where(split == "val")[0]
    te_idx = np.where(split == "test")[0]
    lens = np.array([len(e) for e in enc])
    print(f"train {len(tr_idx):,}  val {len(va_idx):,}  test {len(te_idx):,}  tokens/headline: "
          f"mean {lens.mean():.0f}, p99 {np.percentile(lens, 99):.0f}, truncated {(lens >= args.max_len).sum()}",
          flush=True)

    dtype = torch.bfloat16 if args.method == "lora" else torch.float32
    model, kind = load_classifier(args.model, 3 * N_ASSETS, dtype)
    if kind == "seq_cls":
        model.config.pad_token_id = tok.pad_token_id
        model.config.get_text_config().pad_token_id = tok.pad_token_id     # composite configs (Qwen3.5)
    if args.grad_ckpt:
        (model if kind == "seq_cls" else model.decoder).gradient_checkpointing_enable()
    if args.method == "lora":
        model = add_lora(model, kind, args)
    model.to(device).train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(tr_idx) // args.batch
    total = args.max_steps or int(math.ceil(args.epochs * steps_per_epoch))
    warm = max(1, int(args.warmup * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))

    log, best, bad, t0 = [], {"val_loss": float("inf"), "step": -1}, 0, time.time()
    best_state = None
    val_ids = news["id"].to_numpy()[va_idx]

    def run_eval(step, epoch):
        nonlocal best, bad, best_state
        p, vloss = predict(model, data, list(va_idx), args.eval_batch, device)
        res, _ = evaluate(to_frame(val_ids, p), news, days, "val")
        nl, dl = res[args.target]["news_level"], res[args.target]["day_level"]
        row = {"step": step, "epoch": round(epoch, 3), "train_loss": np.nan, "val_loss": vloss}
        for a in ASSETS:
            row[f"val_acc_{a}"], row[f"val_f1_{a}"] = nl[a]["accuracy"], nl[a]["macro_f1"]
            row[f"val_dayacc_{a}"] = dl[a]["accuracy_dir"]
        log.append(row)
        improved = vloss < best["val_loss"] - 1e-4
        if improved:
            best = {"val_loss": vloss, "step": step, "epoch": round(epoch, 3),
                    **{k: v for k, v in row.items() if k.startswith("val_")}}
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.named_parameters() if v.requires_grad}
            bad = 0
        else:
            bad += 1
        print(f"[eval] step {step} ep {epoch:.2f} val_loss {vloss:.4f} "
              + " ".join(f"{a}:acc {nl[a]['accuracy']:.3f} f1 {nl[a]['macro_f1']:.3f} day {dl[a]['accuracy_dir']:.2f}"
                         for a in ASSETS)
              + (" *best*" if improved else f" (no gain {bad}/{args.patience})"), flush=True)

    run_eval(0, 0.0)
    step, run_loss, run_n, stop = 0, 0.0, 0, False
    rng = np.random.default_rng(args.seed)
    while not stop:
        order = rng.permutation(tr_idx)
        for s in range(0, len(order) - args.batch + 1, args.batch):
            x, m, y = data.make(order[s:s + args.batch])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=x.to(device), attention_mask=m.to(device)).logits
            loss = loss_fn(logits, y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            run_loss += loss.item()
            run_n += 1
            epoch = step / steps_per_epoch
            if step % 25 == 0:
                log.append({"step": step, "epoch": round(epoch, 3), "train_loss": run_loss / run_n,
                            "val_loss": np.nan})
                el = time.time() - t0
                print(f"step {step}/{total} ep {epoch:.2f} loss {run_loss / run_n:.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {step / el:.2f} it/s, eta {(total - step) / (step / el) / 60:.0f} min",
                      flush=True)
                run_loss, run_n = 0.0, 0
            if step % args.eval_every == 0 or step == total:
                run_eval(step, epoch)
                pd.DataFrame(log).to_csv(os.path.join(out, "train_log.csv"), index=False)
                if bad >= args.patience:
                    print(f"early stop at step {step}: no better val loss for {bad} evaluations", flush=True)
                    stop = True
            if step >= total or stop:
                stop = True
                break
    train_min = (time.time() - t0) / 60

    log_df = pd.DataFrame(log)
    log_df.to_csv(os.path.join(out, "train_log.csv"), index=False)
    title = f"{short_name(args.model)} {args.method} | variant {args.variant} | target '{args.target}'"
    plot_curves(log_df, out, title)

    # best step -> weights, predictions for val and test (test is predicted once, nothing is tuned on it)
    with torch.no_grad():
        named = dict(model.named_parameters())
        for k, v in best_state.items():
            named[k].copy_(v.to(named[k].device))
    ckpt = os.path.join(out, "checkpoint")
    save_checkpoint(model, kind, tok, ckpt)
    t1 = time.time()
    ev_idx = np.concatenate([va_idx, te_idx])
    p, _ = predict(model, data, list(ev_idx), args.eval_batch, device)
    pred = to_frame(news["id"].to_numpy()[ev_idx], p)
    metrics = report_run(pred, news, days, out, title)
    infer_s = time.time() - t1
    metrics["training"] = {"best": best, "steps_done": step, "steps_planned": total,
                           "early_stopped": step < total, "train_minutes": round(train_min, 1),
                           "inference_seconds_val_test": round(infer_s, 1),
                           "headlines_per_second": round(len(ev_idx) / infer_s, 1)}
    write_json(metrics, os.path.join(out, "metrics.json"))
    write_json({"model": args.model, "released": RELEASED.get(args.model, "unknown"),
                "knowledge_cutoff_note": "release date is an upper bound of the knowledge cutoff; "
                                         f"it must be before test start {TEST_START}",
                "method": args.method, "head": kind, "variant": args.variant, "input_column": text,
                "target": args.target,
                "splits": {"train": f"< {VAL_START}", "val": f"{VAL_START} .. < {TEST_START}",
                           "test": f"{TEST_START} .. 2026-11-17", "day": "calendar day in America/Los_Angeles"},
                "thresholds_pct": THRESHOLDS, "classes": CLASSES, "assets": list(ASSETS),
                "hyperparameters": {k: v for k, v in vars(args).items() if k not in ("model", "variant", "target")},
                "rows": {"train": int(len(tr_idx)), "val": int(len(va_idx)), "test": int(len(te_idx))},
                "device": torch.cuda.get_device_name(0), "torch": torch.__version__},
               os.path.join(out, "config.json"))
    dl = metrics["test"][args.target]["day_level"]
    print(f"done in {train_min:.1f} min (best step {best['step']}, val loss {best['val_loss']:.4f}); "
          f"test days ({args.target}): " + ", ".join(f"{a} {dl[a]['correct_dir']}/{dl[a]['days']} "
                                                     f"p={dl[a]['p_value_binomial']:.3f} "
                                                     f"centered {dl[a]['correct_dir_centered']}" for a in ASSETS),
          flush=True)


if __name__ == "__main__":
    main()
