#!/usr/bin/env python3
"""
Builds the dataset for the local fine-tune (FINETUNE_PLAN.md, sections 2-4).

experiments/data/news.csv - one row per unique headline:
  id, time (UTC as in the source), day (news day in Los Angeles time), date_only, split,
  text_a (headline), text_b (headline + annotation columns),
  r_<asset>_same / y_<asset>_same - 24 h return from the news time, % and class (target of the plan),
  r_<asset>_next / y_<asset>_next - 24 h return starting when the news day is over (honest forecast).
  Labels whose price window reaches into the next split are blanked (purge), so val/test prices
  never leak into train/val.
experiments/data/days.csv - per news day: r_<asset>_same (the day's own move), r_<asset>_next (next day).

Run: python3 ft_data.py
"""

import os

import numpy as np
import pandas as pd

from ft_common import (ANNOT_NUM, ASSETS, DATA_DIR, LAST_DAY, NEWS_FILES, PRICE_FILE, TARGETS, TEST_START, TZ,
                       VAL_START, day_start_utc, rcol, to_class, ycol)

DAY = pd.Timedelta("24h")


def load_prices():
    p = pd.read_csv(PRICE_FILE, sep=";", usecols=["time_utc"] + list(ASSETS.values()))
    p["time"] = pd.to_datetime(p["time_utc"])
    return p.drop(columns="time_utc").sort_values("time").reset_index(drop=True)


def price_at(prices, times):
    """Price of the first 30-min candle not earlier than each time (max 2 h later), in the input order."""
    q = pd.DataFrame({"time": pd.to_datetime(times), "_i": np.arange(len(times))}).sort_values("time")
    m = pd.merge_asof(q, prices, on="time", direction="forward", tolerance=pd.Timedelta("2h"))
    return m.sort_values("_i").reset_index(drop=True)


def text_b(r):
    def s(v):
        return "" if pd.isna(v) else str(v).replace("|", ", ")
    return (f"{r['news']}\n"
            f"sentiment: {r['sentiment']} ({r['sentiment_score']:+.2f}); globality {r['globality']:.2f}; "
            f"clickbait {r['clickbait']:.2f}; crypto impact {r['crypto_impact']:+.2f}; "
            f"gold impact {r['gold_impact']:+.2f}; economy impact {r['economy_impact']:+.2f}; "
            f"subjects: {s(r['subjects'])}; actors: {s(r['actors'])}")


def main():
    raw = pd.concat([pd.read_csv(p, sep=";", index_col=0, encoding="utf-8-sig") for p in NEWS_FILES])
    raw.index.name = "id"
    raw = raw.reset_index()
    raw["time"] = pd.to_datetime(raw["date"], format="mixed")
    local = raw["time"].dt.tz_localize("UTC").dt.tz_convert(TZ)
    raw["date_only"] = (local.dt.hour == 0) & (local.dt.minute == 0)
    raw["day"] = local.dt.tz_localize(None).dt.normalize()
    raw["split"] = np.where(raw["day"] < VAL_START, "train", np.where(raw["day"] < TEST_START, "val", "test"))
    raw = raw[raw["day"] <= LAST_DAY]
    print(f"headlines: {len(raw):,}, date-only (midnight {TZ}): {raw['date_only'].mean():.1%}")
    print("before dedup:", raw["split"].value_counts().reindex(["train", "val", "test"]).to_dict())

    # one copy of every headline, the earliest one; then the same text cannot be in two splits
    raw["key"] = raw["news"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    news = raw.sort_values(["time", "id"]).drop_duplicates("key").sort_values(["time", "id"]).reset_index(drop=True)
    print(f"after dedup: {len(news):,} (removed {len(raw) - len(news):,}):",
          news["split"].value_counts().reindex(["train", "val", "test"]).to_dict())
    assert news.groupby("key")["split"].nunique().max() == 1

    prices = load_prices()
    bounds = {"train": day_start_utc([VAL_START])[0], "val": day_start_utc([TEST_START])[0],
              "test": pd.Timestamp.max}
    limit = news["split"].map(bounds)

    # target from the plan: from the news time to +24 h
    p0, p1 = price_at(prices, news["time"]), price_at(prices, news["time"] + DAY)
    end_same = news["time"] + DAY
    # honest target: the 24 h after the news day is over
    start_next = pd.Series(day_start_utc(news["day"] + pd.Timedelta("1D")), index=news.index)
    q0, q1 = price_at(prices, start_next), price_at(prices, start_next + DAY)
    end_next = start_next + DAY
    for a, col in ASSETS.items():
        for t, (b, e, end) in {"same": (p0, p1, end_same), "next": (q0, q1, end_next)}.items():
            r = ((e[col] / b[col] - 1) * 100).to_numpy(dtype=float, copy=True)
            r[(end > limit).to_numpy()] = np.nan          # purge: label window crosses into the next split
            news[rcol(a, t)] = r
            news[ycol(a, t)] = to_class(r, a)

    news["text_a"] = news["news"]
    news["text_b"] = news.apply(text_b, axis=1)

    # day table: every news day, its own move and the next day's move
    all_days = pd.date_range(news["day"].min(), news["day"].max(), freq="D")
    st = pd.Series(day_start_utc(all_days))
    d0, d1 = price_at(prices, st), price_at(prices, st + DAY)
    days = pd.DataFrame({"day": all_days, "start_utc": st.to_numpy()})
    cnt = news.groupby("day").size()
    days["n_news"] = days["day"].map(cnt).fillna(0).astype(int)
    days["split"] = np.where(days["day"] < VAL_START, "train", np.where(days["day"] < TEST_START, "val", "test"))
    for a, col in ASSETS.items():
        r = (d1[col] / d0[col] - 1) * 100
        days[rcol(a, "same")] = r.to_numpy()
        days[rcol(a, "next")] = r.shift(-1).to_numpy()
        days[f"r_{a}_prev"] = r.shift(1).to_numpy()

    os.makedirs(DATA_DIR, exist_ok=True)
    keep = (["id", "time", "day", "date_only", "split", "topic", "text_a", "text_b"] + ANNOT_NUM +
            [c for a in ASSETS for t in TARGETS for c in (rcol(a, t), ycol(a, t))])
    news[keep].to_csv(os.path.join(DATA_DIR, "news.csv"), sep=";", index=False, encoding="utf-8-sig")
    days.to_csv(os.path.join(DATA_DIR, "days.csv"), sep=";", index=False)

    # ---------------- checks and summary ----------------
    print("\nclass shares down/flat/up (news level):")
    for t in TARGETS:
        for s in ["train", "val", "test"]:
            part = news[news["split"] == s]
            row = []
            for a in ASSETS:
                y = part[ycol(a, t)].dropna()
                row.append(f"{a} {'/'.join(f'{(y == k).mean():.0%}' for k in range(3))} (n={len(y):,})")
            print(f"  {t:<4} {s:<5} " + "  ".join(row))
    print("\nday level (test):")
    for t in TARGETS:
        td = days[(days["split"] == "test")]
        row = []
        for a in ASSETS:
            r = td[rcol(a, t)].dropna()
            row.append(f"{a}: {len(r)} days, up {(r > 0).mean():.0%}")
        print(f"  {t:<4} " + "  ".join(row))
    print(f"\ntest days {days.loc[days['split'] == 'test', 'day'].min():%Y-%m-%d} .. "
          f"{days.loc[days['split'] == 'test', 'day'].max():%Y-%m-%d}, "
          f"news per month (test): {news[news['split'] == 'test'].groupby(news['day'].dt.to_period('M')).size().to_dict()}")
    print(f"saved {DATA_DIR}/news.csv and days.csv")


if __name__ == "__main__":
    main()
