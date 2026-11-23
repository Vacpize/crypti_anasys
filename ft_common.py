"""
Shared pieces of the local fine-tune experiment (FINETUNE_PLAN.md): constants, data loading,
metrics on the news and day level, plots. Used by ft_data.py, ft_baselines.py, ft_train.py,
ft_zeroshot.py and ft_report.py.

Two targets are evaluated for every model:
  same - the target from the plan (section 4): price change over 24 h starting at the news time.
         Date-only headlines are stamped 07:00/08:00 UTC = midnight in Los Angeles, so this window is
         exactly the day the headline was written in: the headline can *describe* the move.
  next - honest forecast: price change over the 24 h that start when the news day is over.
"""

import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "experiments", "data")
EXP_DIR = os.path.join(HERE, "experiments", "2026-11-23_news-direction")
NEWS_FILES = [os.path.join(HERE, "news_bd1_annotated.csv"), os.path.join(HERE, "news_bd2_annotated.csv")]
PRICE_FILE = os.path.join(HERE, "crypto_price_bd.csv")

TZ = "America/Los_Angeles"          # date-only headlines are midnight in this time zone
VAL_START = "2026-07-18"            # fixed split (FINETUNE_PLAN.md, section 3), news days in TZ
TEST_START = "2026-08-18"
LAST_DAY = "2026-11-18"

ASSETS = {"btc": "btc_usd", "eth": "eth_usd", "gold": "gold_usd_oz"}
THRESHOLDS = {"btc": 1.0, "eth": 1.0, "gold": 0.3}      # %, down <= -t < flat < +t <= up
CLASSES = ["down", "flat", "up"]
TARGETS = ["same", "next"]
ANNOT_NUM = ["sentiment_score", "globality", "clickbait", "crypto_impact", "gold_impact", "economy_impact"]
# months for the per-month breakdown on test (plan, section 7): Aug and Sep together
MONTH_GROUPS = {"2026-08..09": ("2026-08-01", "2026-09-30"), "2026-10": ("2026-10-01", "2026-10-31"),
                "2026-11": ("2026-11-01", "2026-11-30")}


def to_class(r, asset):
    """Return in % -> 0 down / 1 flat / 2 up; NaN stays NaN."""
    t = THRESHOLDS[asset]
    r = np.asarray(r, dtype=float)
    out = np.where(r <= -t, 0, np.where(r >= t, 2, 1)).astype(float)
    out[np.isnan(r)] = np.nan
    return out


def day_start_utc(days):
    """Midnight of the given TZ days, as naive UTC timestamps."""
    idx = pd.DatetimeIndex(pd.to_datetime(days))
    return idx.tz_localize(TZ).tz_convert("UTC").tz_localize(None)


def load_dataset():
    news = pd.read_csv(os.path.join(DATA_DIR, "news.csv"), sep=";", encoding="utf-8-sig",
                       parse_dates=["time", "day"])
    days = pd.read_csv(os.path.join(DATA_DIR, "days.csv"), sep=";", parse_dates=["day"])
    return news, days


def ycol(asset, target):
    return f"y_{asset}_{target}"


def rcol(asset, target):
    return f"r_{asset}_{target}"


def pcols(asset):
    return [f"p_{asset}_{c}" for c in CLASSES]


# ----------------------------- METRICS -------------------------------------


def _binom_p(k, n):
    from scipy.stats import binomtest
    return float(binomtest(int(k), int(n), 0.5, alternative="greater").pvalue) if n else float("nan")


def _corr(x, y):
    from scipy.stats import pearsonr, spearmanr
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        return float("nan"), float("nan")
    return float(pearsonr(x[m], y[m])[0]), float(spearmanr(x[m], y[m])[0])


def news_metrics(pred, news, target):
    """pred: id + p_<asset>_<class>. Accuracy, macro-F1, confusion matrix per asset (rows = fact)."""
    from sklearn.metrics import confusion_matrix, f1_score, log_loss
    df = news[["id"] + [ycol(a, target) for a in ASSETS]].merge(pred, on="id")
    out = {}
    for a in ASSETS:
        d = df.dropna(subset=[ycol(a, target)])
        y = d[ycol(a, target)].astype(int).to_numpy()
        p = d[pcols(a)].to_numpy()
        yhat = p.argmax(1)
        q = np.clip(p, 1e-4, 1)
        out[a] = {
            "n": int(len(d)),
            "accuracy": round(float((yhat == y).mean()), 4),
            "macro_f1": round(float(f1_score(y, yhat, average="macro", labels=[0, 1, 2])), 4),
            "log_loss": round(float(log_loss(y, q / q.sum(1, keepdims=True), labels=[0, 1, 2])), 4),
            "fact_share": [round(float((y == k).mean()), 3) for k in range(3)],
            "pred_share": [round(float((yhat == k).mean()), 3) for k in range(3)],
            "confusion": confusion_matrix(y, yhat, labels=[0, 1, 2]).tolist(),
        }
    return out


def class_mean_returns(news, target):
    """Mean return of each class on train: turns class probabilities into an expected return in %."""
    tr = news[news["split"] == "train"]
    return {a: [float(tr.loc[tr[ycol(a, target)] == k, rcol(a, target)].mean()) for k in range(3)]
            for a in ASSETS}


def daily_table(pred, news, days, target, class_means, thresholds=None):
    """Average class probabilities over all headlines of a day -> one forecast per day and asset.
    Two direction rules: raw "up if P(up) > P(down)" and centered "up if signal > threshold", where the
    threshold is the median daily signal on val (removes the constant bias of a model towards up or down)."""
    df = news[["id", "day"]].merge(pred, on="id")
    g = df.groupby("day")[[c for a in ASSETS for c in pcols(a)]].mean()
    g["n_news"] = df.groupby("day").size()
    g = g.reset_index().merge(days[["day"] + [rcol(a, target) for a in ASSETS]], on="day", how="left")
    rows = []
    for a in ASSETS:
        p = g[pcols(a)].to_numpy()
        r = g[rcol(a, target)].to_numpy()
        d = pd.DataFrame({"day": g["day"], "asset": a, "n_news": g["n_news"],
                          "p_down": p[:, 0], "p_flat": p[:, 1], "p_up": p[:, 2]})
        d["signal"] = d["p_up"] - d["p_down"]
        thr = (thresholds or {}).get(a, float(d["signal"].median()))
        d["pred_class"] = [CLASSES[k] for k in p.argmax(1)]
        d["pred_dir"] = np.where(d["p_up"] > d["p_down"], "up", "down")
        d["threshold"] = thr
        d["pred_dir_centered"] = np.where(d["signal"] > thr, "up", "down")
        d["pred_return"] = p @ np.array(class_means[a])
        d["fact_return"] = r
        fc = to_class(r, a)
        d["fact_class"] = [CLASSES[int(k)] if np.isfinite(k) else "" for k in fc]
        d["fact_dir"] = np.where(r > 0, "up", np.where(r < 0, "down", ""))
        d["correct_dir"] = np.where(d["fact_dir"] == "", np.nan, (d["pred_dir"] == d["fact_dir"]).astype(float))
        d["correct_dir_centered"] = np.where(d["fact_dir"] == "", np.nan,
                                             (d["pred_dir_centered"] == d["fact_dir"]).astype(float))
        d["correct_class"] = np.where(d["fact_class"] == "", np.nan,
                                      (d["pred_class"] == d["fact_class"]).astype(float))
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def signal_thresholds(dailies):
    """{target: {asset: threshold}} of the centered rule, taken from the val daily tables."""
    return {t: {a: float(x["threshold"].iloc[0]) for a, x in d.groupby("asset", sort=False)}
            for t, d in dailies.items()}


def _strategy(pred_dir, fact_return):
    pos = np.where(pred_dir == "up", 1.0, -1.0)
    return round(float((np.prod(1 + pos * fact_return / 100) - 1) * 100), 2)


def day_metrics(daily):
    """Main metrics of the plan: correct up/down days, binomial p, correlation of signal with the fact."""
    out = {}
    for a, d in daily.groupby("asset", sort=False):
        d = d[d["correct_dir"].notna()]
        k, n = int(d["correct_dir"].sum()), int(len(d))
        kc = int(d["correct_dir_centered"].sum())
        pr, sp = _corr(d["signal"].to_numpy(), d["fact_return"].to_numpy())
        res = {
            "days": n, "correct_dir": k, "accuracy_dir": round(k / n, 4) if n else None,
            "p_value_binomial": round(_binom_p(k, n), 4),
            "correct_dir_centered": kc, "accuracy_dir_centered": round(kc / n, 4) if n else None,
            "p_value_centered": round(_binom_p(kc, n), 4),
            "threshold_centered": round(float(d["threshold"].iloc[0]), 5) if n else None,
            "accuracy_3class": round(float(d["correct_class"].mean()), 4),
            "pearson_signal_vs_return": round(pr, 4), "spearman_signal_vs_return": round(sp, 4),
            "mae_pred_return_pct": round(float((d["pred_return"] - d["fact_return"]).abs().mean()), 4),
            "share_pred_up": round(float((d["pred_dir"] == "up").mean()), 3),
            "share_pred_up_centered": round(float((d["pred_dir_centered"] == "up").mean()), 3),
            "share_fact_up": round(float((d["fact_dir"] == "up").mean()), 3),
            "strategy_return_pct": _strategy(d["pred_dir"].to_numpy(), d["fact_return"].to_numpy()),
            "strategy_centered_return_pct": _strategy(d["pred_dir_centered"].to_numpy(),
                                                      d["fact_return"].to_numpy()),
            "hold_return_pct": round(float((np.prod(1 + d["fact_return"] / 100) - 1) * 100), 2),
            "by_month": {},
        }
        for name, (lo, hi) in MONTH_GROUPS.items():
            m = d[(d["day"] >= lo) & (d["day"] <= hi)]
            if len(m):
                pr_m, sp_m = _corr(m["signal"].to_numpy(), m["fact_return"].to_numpy())
                res["by_month"][name] = {"days": int(len(m)), "news": int(m["n_news"].sum()),
                                         "correct_dir": int(m["correct_dir"].sum()),
                                         "accuracy_dir": round(float(m["correct_dir"].mean()), 4),
                                         "correct_dir_centered": int(m["correct_dir_centered"].sum()),
                                         "spearman_signal_vs_return": round(sp_m, 4)}
        out[a] = res
    return out


def evaluate(pred, news, days, split, thresholds=None):
    """All metrics for one prediction set on one split, for both targets.
    thresholds: {target: {asset: x}} for the centered rule (from val); None -> median of this split."""
    sub = news[news["split"] == split]
    pred = pred[pred["id"].isin(sub["id"])]
    res, dailies = {}, {}
    for t in TARGETS:
        daily = daily_table(pred, sub, days, t, class_mean_returns(news, t), (thresholds or {}).get(t))
        res[t] = {"news_level": news_metrics(pred, sub, t), "day_level": day_metrics(daily)}
        dailies[t] = daily
    return res, dailies


# ----------------------------- PLOTS ---------------------------------------


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_pred_vs_fact(daily, title, path):
    plt = _plt()
    fig, axes = plt.subplots(len(ASSETS), 1, figsize=(14, 3.4 * len(ASSETS)), sharex=True)
    for ax, (a, d) in zip(axes, daily.groupby("asset", sort=False)):
        d = d.sort_values("day")
        colors = np.where(d["fact_return"] >= 0, "#2a9d8f", "#e76f51")
        ax.bar(d["day"], d["fact_return"], color=colors, alpha=0.6, width=0.8, label="fact, % over 24 h")
        ax.set_ylabel(f"{a.upper()} return, %")
        ax.axhline(0, color="black", lw=0.6)
        ax2 = ax.twinx()
        ax2.plot(d["day"], d["signal"], color="#264653", lw=1.6, label="forecast P(up) - P(down)")
        wrong = d[d["correct_dir"] == 0]
        ax2.scatter(wrong["day"], wrong["signal"], marker="x", color="#d62828", s=30, zorder=5,
                    label="wrong direction")
        lim = max(0.05, float(np.nanmax(np.abs(d["signal"]))), abs(float(d["threshold"].iloc[0]))) * 1.15
        ax2.set_ylim(-lim, lim)
        ax2.axhline(0, color="#264653", lw=0.4, ls=":")
        ax2.axhline(float(d["threshold"].iloc[0]), color="#f4a261", lw=1, ls="--",
                    label="threshold of the centered rule (val median)")
        ax2.set_ylabel("signal")
        if a == list(ASSETS)[0]:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    axes[0].set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_errors_and_strategy(daily, title, path):
    plt = _plt()
    fig, axes = plt.subplots(2, len(ASSETS), figsize=(15, 7.5))
    for j, (a, d) in enumerate(daily.groupby("asset", sort=False)):
        d = d.sort_values("day").dropna(subset=["fact_return"])
        err = d["pred_return"] - d["fact_return"]
        axes[0, j].hist(err, bins=30, color="#457b9d")
        axes[0, j].axvline(0, color="black", lw=0.8)
        axes[0, j].set_title(f"{a.upper()}: forecast - fact, % (MAE {err.abs().mean():.2f})")
        pos = np.where(d["pred_dir"] == "up", 1.0, -1.0)
        strat = (np.cumprod(1 + pos * d["fact_return"] / 100) - 1) * 100
        hold = (np.cumprod(1 + d["fact_return"] / 100) - 1) * 100
        axes[1, j].plot(d["day"], strat, label="buy if up / sell if down", color="#2a9d8f")
        axes[1, j].plot(d["day"], hold, label="hold", color="#6c757d", ls="--")
        axes[1, j].axhline(0, color="black", lw=0.5)
        axes[1, j].set_title(f"{a.upper()}: cumulative return, %")
        axes[1, j].tick_params(axis="x", rotation=45)
        axes[1, j].legend(fontsize=8)
    fig.suptitle(title + " (illustration, not a trading advice)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_confusions(news_level, title, path):
    plt = _plt()
    fig, axes = plt.subplots(1, len(ASSETS), figsize=(12, 3.8))
    for ax, a in zip(axes, ASSETS):
        cm = np.array(news_level[a]["confusion"], dtype=float)
        norm = cm / cm.sum(1, keepdims=True).clip(min=1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for k in range(3):
                ax.text(k, i, f"{int(cm[i, k])}\n{norm[i, k]:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if norm[i, k] > 0.6 else "black")
        ax.set_xticks(range(3), CLASSES)
        ax.set_yticks(range(3), CLASSES)
        ax.set_xlabel("forecast")
        ax.set_ylabel("fact")
        ax.set_title(f"{a.upper()} acc {news_level[a]['accuracy']:.1%}, F1 {news_level[a]['macro_f1']:.2f}")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ----------------------------- OUTPUT --------------------------------------


def save_predictions(pred, news, path):
    """Wide table: one row per headline, probabilities, forecast and fact for both targets."""
    df = news[["id", "day", "split"]].merge(pred, on="id")
    for a in ASSETS:
        df[f"pred_{a}"] = [CLASSES[k] for k in df[pcols(a)].to_numpy().argmax(1)]
    facts = news[["id"] + [ycol(a, t) for a in ASSETS for t in TARGETS]].copy()
    for a in ASSETS:
        for t in TARGETS:
            facts[f"fact_{a}_{t}"] = facts.pop(ycol(a, t)).map(lambda k: CLASSES[int(k)] if k == k else "")
    df = df.merge(facts, on="id")
    df["day"] = df["day"].dt.strftime("%Y-%m-%d")
    df.round(4).to_csv(path, index=False)


def write_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def report_run(pred, news, days, out_dir, title, splits=("val", "test")):
    """Metrics, daily tables, predictions and plots for one model / baseline run."""
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    metrics, thresholds = {}, None
    for split in splits:
        res, dailies = evaluate(pred, news, days, split, thresholds if split != "val" else None)
        if split == "val":
            thresholds = signal_thresholds(dailies)
        metrics[split] = res
        save_predictions(pred[pred["id"].isin(news.loc[news["split"] == split, "id"])], news,
                         os.path.join(out_dir, f"predictions_{split}.csv"))
        if split == "test":
            daily = pd.concat([d.assign(target=t) for t, d in dailies.items()], ignore_index=True)
            daily["day"] = daily["day"].dt.strftime("%Y-%m-%d")
            daily.round(5).to_csv(os.path.join(out_dir, "daily_test.csv"), index=False)
            for t, d in dailies.items():
                plot_pred_vs_fact(d, f"{title} | test, target '{t}'",
                                  os.path.join(out_dir, "plots", f"test_pred_vs_fact_{t}.png"))
                plot_errors_and_strategy(d, f"{title} | test, target '{t}'",
                                         os.path.join(out_dir, "plots", f"test_errors_strategy_{t}.png"))
                plot_confusions(res[t]["news_level"], f"{title} | test news level, target '{t}'",
                                os.path.join(out_dir, "plots", f"test_confusion_{t}.png"))
    return metrics
