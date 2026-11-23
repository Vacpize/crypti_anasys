#!/usr/bin/env python3
"""
Summary of all runs of the experiment (FINETUNE_PLAN.md, section 8): one table and comparison plots for
baselines, fine-tuned models and the zero-shot model; the final model per target is chosen on val
(lowest val log-loss), its test result is the main answer. Runs with a tag (pilot, smoke, try) are skipped.

  python3 ft_report.py          # reads experiments/<exp>/**/metrics.json, writes summary.* and plots/
"""

import glob
import json
import os

import numpy as np
import pandas as pd

from ft_common import ASSETS, EXP_DIR, TARGETS, _binom_p, _plt, write_json

SUCCESS = ("one-sided binomial p < 0.05 for the share of days with the right direction (55/92 days for 'same', "
           "54/91 for 'next') and more right days than the best baseline, for variant A and for variant B")


def k_min(n):
    """Fewest right days out of n with one-sided binomial p < 0.05."""
    return next(k for k in range(n + 1) if _binom_p(k, n) < 0.05)


def dir_size_mb(path):
    return round(sum(os.path.getsize(f) for f in glob.glob(os.path.join(path, "**"), recursive=True)
                     if os.path.isfile(f)) / 1e6, 1) if os.path.isdir(path) else None


def collect():
    rows = []
    for mpath in sorted(glob.glob(os.path.join(EXP_DIR, "**", "metrics.json"), recursive=True)):
        run_dir = os.path.dirname(mpath)
        rel = os.path.relpath(run_dir, EXP_DIR).replace("\\", "/")
        if rel.count("/") != 1:
            continue
        group, run = rel.split("/")
        m = json.load(open(mpath, encoding="utf-8"))
        cfg_path = os.path.join(run_dir, "config.json")
        cfg = json.load(open(cfg_path, encoding="utf-8")) if os.path.exists(cfg_path) else {}
        if group == "baselines":
            name, trained_for = run.rsplit("_", 1)
            base = {"kind": "baseline", "model": name, "variant": "-", "method": "-", "trained_for": trained_for}
            eval_targets = [trained_for]
        elif run.startswith("zeroshot"):
            if run != "zeroshot":
                continue
            base = {"kind": "zero-shot", "model": cfg.get("model", group), "variant": "A", "method": "zero-shot",
                    "trained_for": "-"}
            eval_targets = list(TARGETS)
        else:
            parts = run.split("_")
            if len(parts) != 2:
                continue                       # tagged run: pilot / smoke
            base = {"kind": "fine-tune", "model": cfg.get("model", group), "variant": parts[0],
                    "method": cfg.get("method", "?"), "trained_for": parts[1]}
            eval_targets = [parts[1]]
        tr, inf = m.get("training", {}), m.get("inference", {})
        for t in eval_targets:
            row = {**base, "target": t, "run": rel}
            dl, nl = m["test"][t]["day_level"], m["val"][t]["news_level"]
            vdl = m["val"][t]["day_level"]
            for a in ASSETS:
                d = dl[a]
                row.update({f"{a}_days": d["days"], f"{a}_correct": d["correct_dir"], f"{a}_acc": d["accuracy_dir"],
                            f"{a}_p": d["p_value_binomial"], f"{a}_correct_c": d["correct_dir_centered"],
                            f"{a}_acc_c": d["accuracy_dir_centered"], f"{a}_p_c": d["p_value_centered"],
                            f"{a}_rho": d["spearman_signal_vs_return"], f"{a}_strategy": d["strategy_return_pct"],
                            f"{a}_hold": d["hold_return_pct"], f"{a}_val_logloss": nl[a]["log_loss"],
                            f"{a}_val_dayacc": vdl[a]["accuracy_dir"],
                            f"{a}_test_news_acc": m["test"][t]["news_level"][a]["accuracy"],
                            f"{a}_test_news_f1": m["test"][t]["news_level"][a]["macro_f1"]})
            row["val_logloss_mean"] = round(float(np.mean([nl[a]["log_loss"] for a in ASSETS])), 4)
            row["train_minutes"] = tr.get("train_minutes")
            row["best_epoch"] = tr.get("best", {}).get("epoch")
            row["headlines_per_second_gpu"] = tr.get("headlines_per_second", inf.get("headlines_per_second"))
            row["checkpoint_mb"] = dir_size_mb(os.path.join(run_dir, "checkpoint"))
            row["released"] = cfg.get("released")
            rows.append(row)
    return pd.DataFrame(rows)


def label(r):
    if r["kind"] == "baseline":
        return f"baseline: {r['model']}"
    name = r["model"].split("/")[-1]
    name = {"finbert": "FinBERT"}.get(name, name)
    return f"{name} zero-shot" if r["kind"] == "zero-shot" else f"{name} {r['variant']}"


def plot_comparison(df, t, path):
    plt = _plt()
    d = df[df["target"] == t].copy()
    d["label"] = d.apply(label, axis=1)
    x = np.arange(len(d))
    w = 0.26
    colors = {"btc": "#f4a261", "eth": "#6d597a", "gold": "#e9c46a"}
    fig, axes = plt.subplots(3, 1, figsize=(max(10, 0.9 * len(d) + 3), 11), sharex=True)
    n_days = int(d["btc_days"].iloc[0])
    panels = [("acc", "correct days, rule P(up) > P(down)"), ("acc_c", "correct days, centered rule (val median)"),
              ("rho", "Spearman: day signal vs actual return")]
    for ax, (key, title) in zip(axes, panels):
        for k, a in enumerate(ASSETS):
            ax.bar(x + (k - 1) * w, d[f"{a}_{key}"], width=w, color=colors[a], label=a.upper())
        if key == "rho":
            ax.axhline(0, color="black", lw=0.8)
        else:
            ax.axhline(0.5, color="black", lw=0.8, ls=":", label="coin flip")
            ax.axhline(k_min(n_days) / n_days, color="#d62828", lw=1, ls="--",
                       label=f"p < 0.05: {k_min(n_days)}/{n_days}")
            ax.set_ylim(0.3, 0.7)
        ax.set_title(title)
        ax.legend(fontsize=8, ncol=5, loc="upper left")
    axes[-1].set_xticks(x, d["label"], rotation=35, ha="right")
    fig.suptitle(f"test, target '{t}' ({n_days} days)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def md_table(df, t):
    d = df[df["target"] == t]
    head = ("| model | kind | " + " | ".join(f"{a.upper()} days (p) | {a.upper()} centered | {a.upper()} ρ"
                                             for a in ASSETS) + " | val log-loss |")
    sep = "|" + "---|" * (3 + 3 * len(ASSETS))
    lines = [head, sep]
    for _, r in d.iterrows():
        cells = [label(r), r["kind"]]
        for a in ASSETS:
            cells += [f"{r[f'{a}_correct']}/{r[f'{a}_days']} ({r[f'{a}_p']:.2f})",
                      f"{r[f'{a}_correct_c']}/{r[f'{a}_days']} ({r[f'{a}_p_c']:.2f})", f"{r[f'{a}_rho']:+.2f}"]
        cells.append(f"{r['val_logloss_mean']:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    df = collect()
    if df.empty:
        raise SystemExit("no runs found")
    order = {"baseline": 0, "zero-shot": 1, "fine-tune": 2}
    df = df.sort_values(["target", "kind", "model", "variant"], key=lambda s: s.map(order) if s.name == "kind" else s)
    df.to_csv(os.path.join(EXP_DIR, "summary.csv"), index=False)
    os.makedirs(os.path.join(EXP_DIR, "plots"), exist_ok=True)

    summary = {"success_criterion": SUCCESS,
               "selection": "final model per target and variant = fine-tuned run with the lowest mean val "
                            "log-loss (test is not used for the choice)", "final": {}}
    md = []
    for t in TARGETS:
        if not (df["target"] == t).any():
            continue
        plot_comparison(df, t, os.path.join(EXP_DIR, "plots", f"comparison_test_{t}.png"))
        md += [f"### Target '{t}'", "", md_table(df, t), ""]
        bl = df[(df["target"] == t) & (df["kind"] == "baseline")]
        for v in ("A", "B"):
            ft = df[(df["target"] == t) & (df["kind"] == "fine-tune") & (df["variant"] == v)]
            if not len(ft):
                continue
            best = ft.loc[ft["val_logloss_mean"].idxmin()]
            res = {}
            for a in ASSETS:
                n, k, kc = int(best[f"{a}_days"]), int(best[f"{a}_correct"]), int(best[f"{a}_correct_c"])
                b_raw = int(bl[f"{a}_correct"].max()) if len(bl) else 0
                b_c = int(bl[f"{a}_correct_c"].max()) if len(bl) else 0
                res[a] = {"correct_dir": f"{k}/{n}", "p": best[f"{a}_p"],
                          "correct_dir_centered": f"{kc}/{n}", "p_centered": best[f"{a}_p_c"],
                          "spearman": best[f"{a}_rho"], "best_baseline": f"{b_raw}/{n}",
                          "best_baseline_centered": f"{b_c}/{n}",
                          "success": bool(best[f"{a}_p"] < 0.05 and k > b_raw),
                          "success_centered": bool(best[f"{a}_p_c"] < 0.05 and kc > b_c)}
            summary["final"][f"{t}_{v}"] = {"run": best["run"], "model": best["model"], "variant": v,
                                            "val_logloss_mean": best["val_logloss_mean"], "test": res}
    summary["runs"] = df.to_dict(orient="records")
    write_json(summary, os.path.join(EXP_DIR, "metrics.json"))
    with open(os.path.join(EXP_DIR, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n".join(md))
    print(json.dumps(summary["final"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
