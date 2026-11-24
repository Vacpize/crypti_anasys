#!/usr/bin/env python3
"""
Adds price-state columns to the news files: what the market looked like right before the headline.
Only prices up to the news time are used (last candle <= t), so nothing leaks the future and the
columns can be used as model inputs. Assets: btc, eth, gold (from crypto_price_bd.csv).

inertia_<asset>        is the price keeping its direction over the last 24h, -1..1
                       = (P(t) - P(t-24h)) / sum of |30-min price changes| (signed Kaufman efficiency ratio)
                       +1 steady rise, ~0.5 rising with pullbacks, 0 sideways/choppy, -1 steady fall
volatility_<asset>     how strongly the price moves, from the last 24h, in % per day
volatility_<asset>_7d  same from the last 7 days (steadier; the best predictor of tomorrow's move size)
                       = std of 30-min returns * sqrt(48); e.g. BTC 3 means a typical day moves ~3%
                       Candles with no price change (gold on weekends) are skipped.

A value is empty when the window has too few price changes (fewer than 6 per day of window):
gold before ~2020 has only daily prices, and gold is flat over weekends. The first 7 days of data
have no full history either.

Writes the columns into news_bd*_annotated.csv in place (existing columns with these names are replaced).
annotate_news.py collect rebuilds those files from scratch, so run this script again after it.

Run: python3 add_price_features.py
"""

import argparse
import glob

import numpy as np
import pandas as pd

ASSETS = {"btc": "btc_usd", "eth": "eth_usd", "gold": "gold_usd_oz"}
MIN_MOVES_PER_DAY = 6        # need at least this many 30-min price changes per day of window, else empty
CANDLES_PER_DAY = 48
# column suffix -> (feature, window)
FEATURES = {"inertia": ("inertia", "24h"), "volatility": ("volatility", "24h"),
            "volatility_7d": ("volatility", "7D")}


def enough_moves(step, window):
    moves = (step != 0).astype(float).rolling(window).sum()
    return moves >= MIN_MOVES_PER_DAY * pd.Timedelta(window) / pd.Timedelta("1D")


def inertia(s, window):
    net = s - s.shift(freq=window).reindex(s.index, method="ffill")
    step = s.diff()
    path = step.abs().rolling(window).sum()
    er = (net / path).where(path > 0, 0.0)
    return er.where(enough_moves(step, window)).clip(-1, 1).round(3)


def volatility(s, window):
    step = s.diff()
    ret = np.log(s).diff().where(step != 0)          # skip candles where the market did not move
    vol = ret.rolling(window, min_periods=2).std() * np.sqrt(CANDLES_PER_DAY) * 100
    return vol.where(enough_moves(step, window)).round(3)


def feature_table(prices):
    """Per-candle features, columns named <feature>_<asset>[_7d], indexed by time."""
    p = prices.set_index("time")[list(ASSETS.values())]
    out = pd.DataFrame(index=p.index)
    funcs = {"inertia": inertia, "volatility": volatility}
    for a, col in ASSETS.items():
        for name, (feat, window) in FEATURES.items():
            c = f"{feat}_{a}" + ("_7d" if name.endswith("_7d") else "")
            v = funcs[feat](p[col], window)
            v[v.index < p.index[0] + pd.Timedelta(window)] = np.nan   # no full history yet
            out[c] = v
    return out.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--news", nargs="*", default=sorted(glob.glob("news_bd*_annotated.csv")))
    ap.add_argument("--prices", default="crypto_price_bd.csv")
    args = ap.parse_args()
    if not args.news:
        raise SystemExit("No news_bd*_annotated.csv files found.")

    prices = pd.read_csv(args.prices, sep=";", usecols=["time_utc"] + list(ASSETS.values()))
    prices["time"] = pd.to_datetime(prices["time_utc"])
    prices = prices.drop(columns="time_utc").sort_values("time")
    feats = feature_table(prices)
    cols = [c for c in feats.columns if c != "time"]

    for path in args.news:
        df = pd.read_csv(path, sep=";", index_col=0, encoding="utf-8-sig")
        t = pd.DataFrame({"time": pd.to_datetime(df["date"], format="mixed"), "row": range(len(df))})
        m = pd.merge_asof(t.sort_values("time"), feats, on="time", direction="backward",
                          tolerance=pd.Timedelta("2h")).sort_values("row")
        for c in cols:
            df[c] = m[c].to_numpy()
        df.to_csv(path, sep=";", encoding="utf-8-sig")
        filled = ", ".join(f"{c} {df[c].notna().sum():,}" for c in cols if c.endswith("btc") or "gold" in c)
        print(f"{path} ({len(df):,} rows) filled: {filled}")


if __name__ == "__main__":
    main()
