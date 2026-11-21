#!/usr/bin/env python3
"""
Joins annotated news with BTC / ETH / gold prices and shows how prices moved after the news.

For every headline: price at publication time and 1h / 4h / 24h later -> return in %.
Then average return and average absolute move grouped by sentiment, subject, globality, clickbait.

Run:  python analyze_news_prices.py                 # uses *_annotated.csv
      python analyze_news_prices.py --save joined.csv   # also saves the per-headline table
"""

import argparse
import glob

import pandas as pd

ASSETS = {"btc": "btc_usd", "eth": "eth_usd", "gold": "gold_usd_oz"}
HORIZONS = {"1h": "1h", "4h": "4h", "24h": "24h"}


def load_news(paths):
    df = pd.concat([pd.read_csv(p, sep=";", index_col=0, encoding="utf-8-sig") for p in paths])
    df["time"] = pd.to_datetime(df["date"], format="mixed")
    return df.sort_values("time").reset_index(drop=True)


def load_prices(path):
    p = pd.read_csv(path, sep=";", usecols=["time_utc"] + list(ASSETS.values()))
    p["time"] = pd.to_datetime(p["time_utc"])
    return p.drop(columns="time_utc").sort_values("time").reset_index(drop=True)


def add_returns(news, prices):
    """Adds ret_<asset>_<h> (%) = price(t+h) / price(t) - 1, price taken at the first candle >= t."""
    base = pd.merge_asof(news[["time"]], prices, on="time", direction="forward",
                         tolerance=pd.Timedelta("2h"))
    for h, delta in HORIZONS.items():
        later = news[["time"]].copy()
        later["time"] += pd.Timedelta(delta)
        fut = pd.merge_asof(later, prices, on="time", direction="forward",
                            tolerance=pd.Timedelta("2h"))
        for a, col in ASSETS.items():
            news[f"ret_{a}_{h}"] = (fut[col] / base[col] - 1) * 100
    return news


def table(df, key, h, min_n=200):
    cols = [f"ret_{a}_{h}" for a in ASSETS]
    g = df.groupby(key, observed=True)
    out = g[cols].mean().round(3)
    out.columns = [f"{a}_mean%" for a in ASSETS]
    absm = g[cols].agg(lambda s: s.abs().mean()).round(3)
    absm.columns = [f"{a}_|move|%" for a in ASSETS]
    out = out.join(absm)
    out.insert(0, "n", g.size())
    return out[out["n"] >= min_n].sort_values("n", ascending=False)


def daily_table(news, prices):
    """Day-level view: news features of day D vs price change the day before, the same day, the next day.

    Most headlines only have a date (time is always 07:00), and all headlines of one day share the same
    price move, so the day is the honest unit of analysis. Values are Spearman rank correlations.
    """
    n = news.dropna(subset=["globality"]).copy()
    n["day"] = n["time"].dt.floor("D")
    n["neg"] = n["sentiment"] == "negative"
    n["hi"] = n["globality"] >= 0.6
    n["ws"] = n["sentiment_score"] * n["globality"]
    g = n.groupby("day")
    d = pd.DataFrame({
        "n": g.size(),
        "sentiment": g["sentiment_score"].mean(),
        "sentiment_x_globality": g["ws"].sum() / g["globality"].sum().clip(lower=1e-9),
        "crypto_sentiment": n[n["subjects"].str.contains("crypto")].groupby("day")["sentiment_score"].mean(),
        "share_negative": g["neg"].mean(),
        "globality": g["globality"].mean(),
        "share_global>=0.6": g["hi"].mean(),
        "clickbait": g["clickbait"].mean(),
    })
    px = prices.set_index("time").resample("1D").first()
    for a, col in ASSETS.items():
        r = (px[col].shift(-1) / px[col] - 1) * 100
        d[f"{a}_prev"], d[f"{a}_same"], d[f"{a}_next"], d[f"{a}_|same|"] = r.shift(1), r, r.shift(-1), r.abs()
    d = d[d["n"] >= 5]

    def rank_corr(x, y):
        m = x.notna() & y.notna()
        return round(x[m].rank().corr(y[m].rank()), 3)

    feats = [c for c in d.columns if c != "n" and not c.startswith(tuple(ASSETS))]
    targets = [c for c in d.columns if c.startswith(tuple(ASSETS))]
    out = pd.DataFrame({t: {f: rank_corr(d[f], d[t]) for f in feats} for t in targets})
    return out, len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--news", nargs="*", default=sorted(glob.glob("news_bd*_annotated.csv")))
    ap.add_argument("--prices", default="crypto_price_bd.csv")
    ap.add_argument("--horizon", default="24h", choices=HORIZONS)
    ap.add_argument("--save", help="write the joined per-headline table to this CSV")
    args = ap.parse_args()
    if not args.news:
        raise SystemExit("No *_annotated.csv files found. Run annotate_news.py first.")

    prices = load_prices(args.prices)
    news = add_returns(load_news(args.news), prices)
    h = args.horizon
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(f"{len(news):,} headlines, {news['time'].min()} .. {news['time'].max()}, horizon {h}")
    base_abs = {a: news[f"ret_{a}_{h}"].abs().mean() for a in ASSETS}
    print("baseline |move| %:", {a: round(v, 3) for a, v in base_abs.items()})

    print("\n== by topic ==")
    print(table(news, "topic", h))

    annotated = news.dropna(subset=["globality"]) if "globality" in news else news.iloc[0:0]
    if annotated.empty:
        print("\n(no annotation columns filled yet)")
    else:
        a = annotated.copy()
        print(f"\nannotated rows: {len(a):,}")
        print("\n== by sentiment ==")
        print(table(a, "sentiment", h))
        a["globality_bin"] = pd.cut(a["globality"], [-0.01, 0.2, 0.4, 0.6, 0.8, 1.0])
        print("\n== by globality ==")
        print(table(a, "globality_bin", h, min_n=50))
        a["clickbait_bin"] = pd.cut(a["clickbait"], [-0.01, 0.2, 0.4, 0.6, 1.0])
        print("\n== by clickbait ==")
        print(table(a, "clickbait_bin", h, min_n=50))
        a["subject"] = a["subjects"].fillna("").str.split("|").str[0]
        print("\n== by main subject ==")
        print(table(a, "subject", h))
        print("\n== main subject x sentiment (BTC) ==")
        piv = a.pivot_table(index="subject", columns="sentiment", values=f"ret_btc_{h}",
                            aggfunc="mean").round(3)
        print(piv)
        print("\n== correlation of sentiment_score with returns ==")
        for scope, part in [("all", a), ("globality>=0.6", a[a["globality"] >= 0.6])]:
            c = {x: round(part["sentiment_score"].corr(part[f"ret_{x}_{h}"]), 4) for x in ASSETS}
            print(f"  {scope} (n={len(part):,}): {c}")
        actors = a["actors"].fillna("").str.split("|").explode()
        print("\n== top 25 actors ==")
        print(actors[actors != ""].value_counts().head(25))

    if not annotated.empty:
        corr, n_days = daily_table(news, prices)
        print(f"\n== day level: rank correlation of news features with price change ({n_days:,} days) ==")
        print("   prev = day before the news, same = news day, next = day after, |same| = size of the move")
        print(f"   values within +-{2 / n_days ** 0.5:.3f} are noise")
        print(corr.to_string())

    if args.save:
        news.to_csv(args.save, sep=";", encoding="utf-8-sig")
        print(f"\nsaved {args.save}")


if __name__ == "__main__":
    main()
