
import argparse
import csv
import sys
import time
from datetime import datetime, timezone

import requests

INTERVAL_MS = {
    "1s": 1_000,
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}
BINANCE_START = "2017-08-17"  # first BTCUSDT / ETHUSDT candles
LIMIT = 1000                  # max candles per request


OZ_PER_KG = 1000 / 31.1034768  # troy ounces in one kilogram = 32.1507466
YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


def fetch_gold_daily(symbol, start_ms, end_ms, retries=5):
    """Return {'YYYY-MM-DD': close_usd_per_oz} from Yahoo Finance (daily bars).

    symbol: 'GC=F' (COMEX gold futures, long history) or 'XAUUSD=X' (spot).
    """
    params = {
        "period1": start_ms // 1000 - 7 * 86400,  # a week earlier for forward-fill
        "period2": end_ms // 1000 + 86400,
        "interval": "1d",
    }
    headers = {"User-Agent": "Mozilla/5.0"}  # Yahoo rejects the default python UA
    for attempt in range(retries):
        try:
            r = requests.get(YAHOO_URL.format(symbol=symbol), params=params,
                             headers=headers, timeout=30)
            r.raise_for_status()
            result = r.json()["chart"]["result"][0]
            stamps = result["timestamp"]
            closes = result["indicators"]["quote"][0]["close"]
            out = {}
            for ts, close in zip(stamps, closes):
                if close:
                    day = datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")
                    out[day] = close
            if not out:
                raise ValueError("no gold data returned")
            return out
        except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
            wait = 2 ** attempt * 2
            print(f"  gold request failed ({exc}); retrying in {wait}s...",
                  file=sys.stderr)
            time.sleep(wait)
    print("Could not download gold prices - gold columns will be blank.",
          file=sys.stderr)
    return {}


def to_ms(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_klines(base_url, symbol, interval, start_ms, end_ms, retries=6):
    """Return list of [open_time_ms, close_price] for one window."""
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": LIMIT,
    }
    for attempt in range(retries):
        try:
            r = requests.get(f"{base_url}/api/v3/klines", params=params, timeout=30)
            if r.status_code == 451:
                raise SystemExit(
                    "HTTP 451: Binance is blocked in your region. Try "
                    "--base-url https://data-api.binance.vision")
            if r.status_code in (418, 429):
                wait = int(r.headers.get("Retry-After", 2 ** attempt * 5))
                print(f"  rate limited, waiting {wait}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            if r.status_code == 400:
                # e.g. symbol/interval not available for this period
                return []
            r.raise_for_status()
            return [[row[0], row[4]] for row in r.json()]
        except requests.RequestException as exc:
            wait = 2 ** attempt
            print(f"  request failed ({exc}); retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)
    raise SystemExit("Too many failed requests - aborting.")


def main():
    p = argparse.ArgumentParser(description="BTC/ETH/USDT price CSV from Binance")
    p.add_argument("--interval", choices=INTERVAL_MS, default="30m")
    p.add_argument("--start", default=BINANCE_START, help="YYYY-MM-DD (UTC)")
    p.add_argument("--end", default=None, help="YYYY-MM-DD (UTC), default: now")
    p.add_argument("--output", default="crypto_prices_binance.csv")
    p.add_argument("--base-url", default="https://api.binance.com")
    p.add_argument("--gold-symbol", default="GC=F",
                   help="Yahoo Finance symbol: GC=F (futures) or XAUUSD=X (spot)")
    p.add_argument("--no-gold", action="store_true", help="skip gold columns")
    p.add_argument("--gold-daily-only", action="store_true",
                   help="use only the Yahoo daily close, not PAXGUSDT intraday candles")
    args = p.parse_args()

    step = INTERVAL_MS[args.interval]
    start_ms = max(to_ms(args.start), to_ms(BINANCE_START))
    end_ms = to_ms(args.end) if args.end else int(time.time() * 1000)

    total_windows = max(1, (end_ms - start_ms) // (step * LIMIT) + 1)
    print(f"Interval {args.interval}: about {total_windows * 3:,} requests needed.")
    if total_windows > 20_000:
        print("That is a lot of requests - consider a shorter range or coarser interval.",
              file=sys.stderr)

    gold = {}
    if not args.no_gold:
        print(f"Downloading daily gold prices ({args.gold_symbol}) ...")
        gold = fetch_gold_daily(args.gold_symbol, start_ms, end_ms)
        print(f"  got {len(gold):,} daily gold closes")
    last_gold = None  # forward-filled USD/oz

    # seed forward-fill with the last close before the start date
    start_day = datetime.fromtimestamp(start_ms / 1000, timezone.utc).strftime("%Y-%m-%d")
    earlier = [d for d in gold if d < start_day]
    if earlier:
        last_gold = gold[max(earlier)]

    rows_written = 0
    window = 0
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["time_utc", "unix_time", "btc_usd", "eth_usd", "usdt_usd"]
        if not args.no_gold:
            header += ["gold_usd_oz", "gold_usd_kg", "gold_source"]
        writer.writerow(header)

        cur = start_ms
        while cur < end_ms:
            win_end = min(cur + step * LIMIT - 1, end_ms)

            btc = dict(fetch_klines(args.base_url, "BTCUSDT", args.interval, cur, win_end))
            eth = dict(fetch_klines(args.base_url, "ETHUSDT", args.interval, cur, win_end))
            usdc = dict(fetch_klines(args.base_url, "USDCUSDT", args.interval, cur, win_end))
            paxg = {}
            if not args.no_gold and not args.gold_daily_only:
                # PAXG = tokenized gold, 1 token = 1 troy ounce
                paxg = dict(fetch_klines(args.base_url, "PAXGUSDT", args.interval, cur, win_end))

            for t in sorted(set(btc) | set(eth) | set(usdc)):
                stamp = datetime.fromtimestamp(t / 1000, timezone.utc).strftime(
                    "%Y-%m-%d %H:%M:%S")
                usdc_close = float(usdc[t]) if t in usdc else 0.0
                usdt_rate = 1 / usdc_close if usdc_close > 0 else 1.0  # USD per USDT
                usdt_usd = round(usdt_rate, 6) if usdc_close > 0 else ""

                def to_usd(series):
                    return round(float(series[t]) * usdt_rate, 2) if t in series else ""

                row = [stamp, t // 1000, to_usd(btc), to_usd(eth), usdt_usd]
                if not args.no_gold:
                    day = stamp[:10]
                    if day in gold:
                        last_gold = gold[day]
                    if t in paxg:
                        oz, src = float(paxg[t]) * usdt_rate, "paxg_intraday"
                    elif last_gold is not None:
                        oz, src = last_gold, "yahoo_daily"
                    else:
                        oz, src = None, ""
                    if oz is not None:
                        row += [round(oz, 2), round(oz * OZ_PER_KG, 2), src]
                    else:
                        row += ["", "", ""]
                writer.writerow(row)
                rows_written += 1

            window += 1
            if window % 20 == 0:
                done = datetime.fromtimestamp(win_end / 1000, timezone.utc)
                print(f"  reached {done:%Y-%m-%d %H:%M:%S} ({rows_written:,} rows)")
                f.flush()

            cur = win_end + 1
            time.sleep(0.1)  # stay well under Binance's rate limits

    print(f"\nDone. {rows_written:,} rows written to {args.output}")


if __name__ == "__main__":
    main()