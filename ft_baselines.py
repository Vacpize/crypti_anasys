#!/usr/bin/env python3
"""
Baselines from FINETUNE_PLAN.md (section 7), with the same metrics as the fine-tuned models:
  majority - class shares of train, the same for every headline
  inertia  - class / direction of the last finished day ('same': the day before the news day,
             'next': the news day itself)
  tfidf_lr - TF-IDF of the headline + logistic regression
  annot_lr - logistic regression on the annotation columns only (sentiment_score, *_impact, globality, clickbait)
C of the regressions is picked on val log-loss; test is only predicted.
Each baseline is fitted for the 'same' and the 'next' target -> experiments/<exp>/baselines/<name>_<target>/

Run: python3 ft_baselines.py
"""

import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

from ft_common import (ANNOT_NUM, ASSETS, EXP_DIR, TARGETS, THRESHOLDS, load_dataset, pcols, report_run, rcol,
                       to_class, write_json, ycol)

C_GRID = {"tfidf_lr": [0.1, 0.3, 1.0, 3.0], "annot_lr": [0.01, 0.1, 1.0, 10.0]}


def frame(ids, probs):
    df = pd.DataFrame({"id": ids})
    for a, p in probs.items():
        df[pcols(a)] = p
    return df


def majority(news, target, ev):
    tr = news[news["split"] == "train"]
    probs = {}
    for a in ASSETS:
        share = tr[ycol(a, target)].value_counts(normalize=True).reindex([0, 1, 2]).fillna(0).to_numpy()
        probs[a] = np.tile(share, (len(ev), 1))
    return frame(ev["id"], probs), {}


def inertia(news, days, target, ev):
    src = {"same": "prev", "next": "same"}[target]       # the last day that is over when the window starts
    d = ev[["id", "day"]].merge(days[["day"] + [f"r_{a}_{src}" for a in ASSETS]], on="day", how="left")
    probs = {}
    for a in ASSETS:
        r = d[f"r_{a}_{src}"].to_numpy(dtype=float)
        k = np.nan_to_num(to_class(r, a), nan=1).astype(int)
        p = np.full((len(d), 3), 0.05)
        p[np.arange(len(d)), k] += 0.85
        p[:, 2] += np.where(r > 0, 0.01, 0)                # up/down tie-break by the sign of the move
        p[:, 0] += np.where(r > 0, 0, 0.01)
        probs[a] = p / p.sum(1, keepdims=True)
    return frame(d["id"], probs), {}


def _fit(X_tr, y_tr, X_va, y_va, C):
    m = LogisticRegression(C=C, max_iter=3000)
    m.fit(X_tr, y_tr)
    return C, m, log_loss(y_va, m.predict_proba(X_va), labels=[0, 1, 2])


def regression(news, target, ev, kind):
    tr = news[news["split"] == "train"]
    va = news[news["split"] == "val"]
    if kind == "tfidf_lr":
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True)
        X_tr, X_va, X_ev = vec.fit_transform(tr["text_a"]), vec.transform(va["text_a"]), vec.transform(ev["text_a"])
    else:
        sc = StandardScaler()
        X_tr = sc.fit_transform(tr[ANNOT_NUM])
        X_va, X_ev = sc.transform(va[ANNOT_NUM]), sc.transform(ev[ANNOT_NUM])
    probs, chosen = {}, {}
    for a in ASSETS:
        mt, mv = tr[ycol(a, target)].notna().to_numpy(), va[ycol(a, target)].notna().to_numpy()
        y_tr, y_va = tr[ycol(a, target)][mt].astype(int), va[ycol(a, target)][mv].astype(int)
        fits = Parallel(n_jobs=len(C_GRID[kind]))(
            delayed(_fit)(X_tr[mt], y_tr, X_va[mv], y_va, C) for C in C_GRID[kind])
        C, model, ll = min(fits, key=lambda f: f[2])
        chosen[a] = {"C": C, "val_log_loss": round(ll, 4),
                     "grid": {str(c): round(l, 4) for c, _, l in fits}}
        probs[a] = model.predict_proba(X_ev)
    return frame(ev["id"], probs), chosen


def main():
    news, days = load_dataset()
    ev = news[news["split"].isin(["val", "test"])].reset_index(drop=True)
    summary = {}
    for name in ["majority", "inertia", "tfidf_lr", "annot_lr"]:
        for target in TARGETS:
            if name == "majority":
                pred, info = majority(news, target, ev)
            elif name == "inertia":
                pred, info = inertia(news, days, target, ev)
            else:
                pred, info = regression(news, target, ev, name)
            out = os.path.join(EXP_DIR, "baselines", f"{name}_{target}")
            metrics = report_run(pred, news, days, out, f"baseline {name} (fitted for '{target}')")
            write_json({"baseline": name, "fitted_for_target": target, "thresholds_pct": THRESHOLDS,
                        "choice_on_val": info}, os.path.join(out, "config.json"))
            write_json(metrics, os.path.join(out, "metrics.json"))
            dl = metrics["test"][target]["day_level"]
            summary[f"{name}_{target}"] = {a: f"{dl[a]['correct_dir']}/{dl[a]['days']} "
                                              f"p={dl[a]['p_value_binomial']:.3f} "
                                              f"centered {dl[a]['correct_dir_centered']}/{dl[a]['days']} "
                                              f"rho={dl[a]['spearman_signal_vs_return']}" for a in ASSETS}
            print(f"{name:<9} {target:<4} test days:", summary[f"{name}_{target}"], flush=True)
    write_json(summary, os.path.join(EXP_DIR, "baselines", "summary_test_days.json"))


if __name__ == "__main__":
    main()
