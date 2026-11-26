#!/usr/bin/env python3
"""Evaluate a saved LoRA checkpoint without repeating its training run."""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from ft_common import (ASSETS, CLASSES, EXP_DIR, TEST_START, THRESHOLDS, VAL_START, load_dataset, report_run,
                       write_json)
from ft_train import Batches, LastTokenHead, labels_of, predict, to_frame


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", choices=["A", "B"], required=True)
    ap.add_argument("--target", choices=["same", "next"], required=True)
    ap.add_argument("--checkpoint", required=True, help="directory with the saved PEFT adapter and score.pt")
    ap.add_argument("--max-len", type=int, required=True)
    ap.add_argument("--eval-batch", type=int, default=128)
    ap.add_argument("--attn-implementation", choices=["eager", "sdpa"], default="eager")
    ap.add_argument("--out", help="run directory (defaults to the model/variant_target directory)")
    return ap.parse_args()


def main():
    args = parse_args()
    out = args.out or os.path.join(EXP_DIR, args.model.split("/")[-1].lower(),
                                   f"{args.variant}_{args.target}")
    checkpoint = os.path.abspath(args.checkpoint)
    if not os.path.isfile(os.path.join(checkpoint, "adapter_model.safetensors")):
        raise FileNotFoundError(f"PEFT adapter not found in {checkpoint}")
    if not os.path.isfile(os.path.join(checkpoint, "score.pt")):
        raise FileNotFoundError(f"Gemma classification head not found in {checkpoint}")

    news, days = load_dataset()
    text_col = "text_a" if args.variant == "A" else "text_b"
    tok = AutoTokenizer.from_pretrained(checkpoint)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    encoded = tok(news[text_col].tolist(), truncation=True, max_length=args.max_len,
                  add_special_tokens=True)["input_ids"]
    data = Batches(encoded, labels_of(news, args.target), tok.pad_token_id)
    val_idx = np.where(news["split"].to_numpy() == "val")[0]
    test_idx = np.where(news["split"].to_numpy() == "test")[0]

    started = time.time()
    lm = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                              attn_implementation=args.attn_implementation)
    decoder = PeftModel.from_pretrained(lm.get_decoder(), checkpoint, is_trainable=False)
    model = LastTokenHead(decoder, lm.config.get_text_config().hidden_size, 3 * len(ASSETS))
    model.score.load_state_dict(torch.load(os.path.join(checkpoint, "score.pt"), map_location="cpu",
                                           weights_only=True))
    model.to("cuda").eval()
    del lm

    eval_idx = np.concatenate([val_idx, test_idx])
    probabilities, _ = predict(model, data, list(eval_idx), args.eval_batch, "cuda")
    pred = to_frame(news["id"].to_numpy()[eval_idx], probabilities)
    title = f"{args.model} LoRA checkpoint | variant {args.variant} | target '{args.target}'"
    metrics = report_run(pred, news, days, out, title)
    elapsed = time.time() - started

    train_log_path = os.path.join(out, "train_log.csv")
    train_log = pd.read_csv(train_log_path)
    val_log = train_log.dropna(subset=["val_loss"])
    best = val_log.loc[val_log["val_loss"].idxmin()]
    metrics["training"] = {
        "best": {"step": int(best["step"]), "val_loss": float(best["val_loss"])},
        "steps_done": int(train_log["step"].max()),
        "checkpoint_evaluation_recovered": True,
        "checkpoint_eval_seconds": round(elapsed, 1),
        "attn_implementation": args.attn_implementation,
    }
    write_json(metrics, os.path.join(out, "metrics.json"))
    write_json({
        "model": args.model,
        "method": "lora",
        "head": "last_token",
        "variant": args.variant,
        "input_column": text_col,
        "target": args.target,
        "splits": {"train": f"< {VAL_START}", "val": f"{VAL_START} .. < {TEST_START}",
                   "test": f"{TEST_START} .. 2026-11-17", "day": "calendar day in America/Los_Angeles"},
        "thresholds_pct": THRESHOLDS,
        "assets": list(ASSETS),
        "hyperparameters": {"max_len": args.max_len, "eval_batch": args.eval_batch,
                            "attn_implementation": args.attn_implementation},
        "rows": {"train": int((news["split"] == "train").sum()), "val": int(len(val_idx)),
                 "test": int(len(test_idx))},
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "checkpoint_evaluation_recovered": True,
    }, os.path.join(out, "config.json"))
    print(f"evaluated saved checkpoint into {out} in {elapsed / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
