#!/usr/bin/env python3
"""
Adds price inertia columns to the news files: is the price keeping its direction (steady rise or
steady fall) at the moment the headline was published.

inertia_<asset> = (P(t) - P(t - window)) / sum of |30-min price changes| over that window, range -1..1
   +1   = price rose steadily (every candle up)
   ~0.5 = rising, but with pullbacks
    0   = sideways / choppy, no direction
   -1   = price fell steadily
This is the signed Kaufman efficiency ratio. Only prices up to the news time are used
(last candle <= t), so the value does not leak the future and can be used as a model input.
Assets: btc, eth, gold (from crypto_price_bd.csv). Empty when the window has fewer than 6 price
changes: gold before ~2020 has only daily prices, and the price is flat over weekends.

Writes the columns into news_bd*_annotated.csv in place (existing inertia columns are replaced).
annotate_news.py collect rebuilds those files from scratch, so run this script again after it.

Run: python3 add_inertia.py                 # 24h window
     python3 add_inertia.py --window 7d     # other window, columns get a suffix: inertia_btc_7d
"""

import argparse
import glob

import pandas as pd

ASSETS = {"btc": "btc_usd", "eth": "eth_usd", "gold": "gold_usd_oz"}
MIN_MOVES = 6   # need at least this many 30-min price changes in the window, else empty


def inertia_series(prices, window):
    """Per-candle signed efficiency ratio over the past `window`, indexed by time."""
    p = prices.set_index("time")[list(ASSETS.values())]
    out = pd.DataFrame(index=p.index)
    for a, col in ASSETS.items():
        s = p[col]
        net = s - s.shift(freq=window).reindex(s.index, method="ffill")
        step = s.diff()
        path = step.abs().rolling(window).sum()
        moves = (step != 0).astype(float).rolling(window).sum()
        er = (net / path).where(path > 0, 0.0)
        # too few price changes in the window (gold before 2020 is daily data) -> one step = fake +-1
        er = er.where(moves >= MIN_MOVES)
        out[a] = er.clip(-1, 1).round(3)
    # the first `window` of data has no full history
    out[out.index < p.index[0] + pd.Timedelta(window)] = float("nan")
    return out.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--news", nargs="*", default=sorted(glob.glob("news_bd*_annotated.csv")))
    ap.add_argument("--prices", default="crypto_price_bd.csv")
    ap.add_argument("--window", default="24h", help="pandas offset, e.g. 24h, 3d, 7d")
    args = ap.parse_args()
    if not args.news:
        raise SystemExit("No news_bd*_annotated.csv files found.")

    prices = pd.read_csv(args.prices, sep=";", usecols=["time_utc"] + list(ASSETS.values()))
    prices["time"] = pd.to_datetime(prices["time_utc"])
    prices = prices.drop(columns="time_utc").sort_values("time")
    ser = inertia_series(prices, args.window)
    suffix = "" if args.window == "24h" else f"_{args.window}"
    cols = {a: f"inertia_{a}{suffix}" for a in ASSETS}

    for path in args.news:
        df = pd.read_csv(path, sep=";", index_col=0, encoding="utf-8-sig")
        t = pd.DataFrame({"time": pd.to_datetime(df["date"], format="mixed"), "row": range(len(df))})
        t = t.sort_values("time")
        m = pd.merge_asof(t, ser, on="time", direction="backward", tolerance=pd.Timedelta("2h"))
        m = m.sort_values("row")
        for a, c in cols.items():
            df[c] = m[a].to_numpy()
        df.to_csv(path, sep=";", encoding="utf-8-sig")
        filled = df[cols["btc"]].notna().sum()
        print(f"{path}: {', '.join(cols.values())} filled for {filled:,}/{len(df):,} rows")


if __name__ == "__main__":
    main()
