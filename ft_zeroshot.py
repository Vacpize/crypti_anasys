#!/usr/bin/env python3
"""
Zero-shot comparison (FINETUNE_PLAN.md: "a big model without training"): an instruction model is asked
for every headline and asset whether the price goes down / stays flat / goes up; the probabilities of the
three answer words are read from the logprobs of the first generated token. No training, nothing tuned on test.
Same outputs and metrics as ft_train.py (report_run), plus inference speed.

  bash experiments/spark_run.sh ft_zeroshot.py                        # nvidia/Qwen3.8-27B-NVFP4, val + test
  bash experiments/spark_run.sh ft_zeroshot.py --limit 300 --tag try  # quick check
"""

import argparse
import math
import os
import time

import numpy as np
import pandas as pd

from ft_common import ASSETS, CLASSES, EXP_DIR, TEST_START, THRESHOLDS, load_dataset, pcols, report_run, write_json

# release of the base model (Qwen/Qwen3.8-27B); the NVFP4 copy is a post-training quantization of it
RELEASED = {"nvidia/Qwen3.8-27B-NVFP4": "2026-08-05"}
NAMES = {"btc": "Bitcoin (BTC)", "eth": "Ethereum (ETH)", "gold": "gold (XAU/USD)"}
SYSTEM = "You are a financial analyst. You judge how a news headline moves asset prices."
QUESTION = ("How will the price of {name} most likely change over the next 24 hours after this news? "
            "down = falls by more than {t}%, flat = stays within {t}%, up = rises by more than {t}%. "
            "Answer with one word: down, flat or up.")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="nvidia/Qwen3.8-27B-NVFP4")
    ap.add_argument("--gpu-mem", type=float, default=0.6, help="share of GPU memory for vLLM")
    ap.add_argument("--limit", type=int, help="only the first N headlines of every split (quick check)")
    ap.add_argument("--tag", default="")
    return ap.parse_args()


def class_probs(top):
    """top: {decoded token: logprob} of the first answer token -> probabilities of down / flat / up."""
    lp = {c: -math.inf for c in CLASSES}
    for tok, v in top.items():
        w = tok.strip().lower()
        if w in lp:
            lp[w] = np.logaddexp(lp[w], v)
    floor = min(top.values()) - 1.0          # a class outside the top-k is less likely than all of them
    x = np.array([lp[c] if np.isfinite(lp[c]) else floor for c in CLASSES])
    e = np.exp(x - x.max())
    return e / e.sum(), sum(np.isfinite(lp[c]) for c in CLASSES)


def main():
    args = parse_args()
    from vllm import LLM, SamplingParams

    news, days = load_dataset()
    ev = news[news["split"].isin(["val", "test"])]
    if args.limit:
        ev = ev.groupby("split", group_keys=False).head(args.limit)
    run = "zeroshot" + (f"_{args.tag}" if args.tag else "")
    out = os.path.join(EXP_DIR, args.model.split("/")[-1].lower(), run)
    os.makedirs(out, exist_ok=True)
    print(f"run -> {out}; {len(ev):,} headlines x {len(ASSETS)} assets", flush=True)

    t0 = time.time()
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=1024,
              enable_prefix_caching=True, limit_mm_per_prompt={"image": 0, "video": 0})
    tok = llm.get_tokenizer()
    load_s = time.time() - t0

    def prompt(headline, a):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f'News headline: "{headline}"\n\n'
                                            + QUESTION.format(name=NAMES[a], t=THRESHOLDS[a])}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    prompts = [prompt(h, a) for h in ev["text_a"] for a in ASSETS]     # same headline first -> prefix cache
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
    t1 = time.time()
    outs = llm.generate(prompts, sp)
    gen_s = time.time() - t1

    p = np.zeros((len(ev), len(ASSETS), 3))
    found = np.zeros(len(outs), dtype=int)
    for i, o in enumerate(outs):
        top = {lp.decoded_token: lp.logprob for lp in o.outputs[0].logprobs[0].values()}
        p[i // len(ASSETS), i % len(ASSETS)], found[i] = class_probs(top)
    pred = pd.DataFrame({"id": ev["id"].to_numpy()})
    for k, a in enumerate(ASSETS):
        pred[pcols(a)] = p[:, k, :]
    print(f"generated in {gen_s:.0f} s ({len(ev) / gen_s:.1f} headlines/s); answer words found in top-20: "
          f"all 3 in {np.mean(found == 3):.1%}, none in {np.mean(found == 0):.1%} of prompts", flush=True)

    title = f"{args.model.split('/')[-1]} zero-shot (no training)"
    metrics = report_run(pred, news, days, out, title)
    metrics["inference"] = {"load_seconds": round(load_s, 1), "generate_seconds": round(gen_s, 1),
                            "headlines": int(len(ev)), "prompts": len(prompts),
                            "headlines_per_second": round(len(ev) / gen_s, 1),
                            "share_prompts_all_words_in_top20": round(float(np.mean(found == 3)), 4)}
    write_json(metrics, os.path.join(out, "metrics.json"))
    write_json({"model": args.model, "released": RELEASED.get(args.model, "unknown"),
                "knowledge_cutoff_note": f"release date is an upper bound of the knowledge cutoff; "
                                         f"it must be before test start {TEST_START}",
                "method": "zero-shot, logprobs of the first answer token, thinking off", "input_column": "text_a",
                "system_prompt": SYSTEM, "question": QUESTION, "asset_names": NAMES, "thresholds_pct": THRESHOLDS,
                "classes": CLASSES, "gpu_memory_utilization": args.gpu_mem, "limit": args.limit},
               os.path.join(out, "config.json"))
    for t in ("next", "same"):
        dl = metrics["test"][t]["day_level"]
        print(f"test days ({t}): " + ", ".join(f"{a} {dl[a]['correct_dir']}/{dl[a]['days']} "
                                              f"centered {dl[a]['correct_dir_centered']} "
                                              f"rho={dl[a]['spearman_signal_vs_return']}" for a in ASSETS), flush=True)


if __name__ == "__main__":
    main()
