import argparse
import csv
import json
import os
import random
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import requests

# ----------------------------- НАСТРОЙКИ -----------------------------------

# Поисковые запросы по темам (синтаксис Google News: OR, кавычки, скобки)
TOPICS = {
    "weather": '(weather OR storm OR hurricane OR tornado OR "heat wave" OR blizzard OR flood OR forecast)',
    "us_politics": '("US politics" OR Congress OR "White House" OR Senate OR "Supreme Court" OR election)',
    "us_market": '("stock market" OR "Wall Street" OR "S&P 500" OR Nasdaq OR "Dow Jones" OR "Federal Reserve")',
    "world_politics": '("world politics" OR NATO OR "United Nations" OR sanctions OR diplomacy OR summit)',
    "tech_ai": '("artificial intelligence" OR "machine learning" OR ChatGPT OR OpenAI OR "large language model")',
    "tech_hardware": '(semiconductor OR chip OR Nvidia OR Apple OR iPhone OR "quantum computing")',
    "tech_cyber": '(cybersecurity OR "data breach" OR ransomware OR hacking)',
    "bitcoin": '(bitcoin OR BTC OR "bitcoin ETF" OR halving OR "bitcoin price")',
    "crypto": '(cryptocurrency OR crypto OR ethereum OR blockchain OR stablecoin OR Binance OR Coinbase)',
    "finance": '(finance OR banking OR "interest rates" OR "central bank" OR bonds OR "hedge fund" OR IPO)',
    "inflation": '(inflation OR CPI OR "consumer prices" OR deflation OR "cost of living")',
    "technology": '(technology OR "artificial intelligence" OR semiconductor OR smartphone OR software OR startup)',
    "taxes": '(tax OR taxes OR "tax reform" OR "tax haven" OR "corporate tax" OR "tax evasion" OR "global minimum tax")',
    "uae_politics": '(UAE OR "United Arab Emirates" OR "Abu Dhabi" OR Dubai) AND (government OR politics OR sheikh OR diplomacy OR minister)',
    "uae_market": '(UAE OR "United Arab Emirates" OR Dubai OR "Abu Dhabi") AND (economy OR market OR "stock exchange" OR ADX OR DFM OR "real estate" OR oil)',
    "turkey_politics": '(Turkey OR Türkiye OR Erdogan OR Ankara) AND (politics OR government OR parliament OR election OR diplomacy)',
    "turkey_market": '(Turkey OR Türkiye OR Turkish) AND (economy OR lira OR "Borsa Istanbul" OR inflation OR "central bank" OR exports)',
    "cybersecurity": '(cybersecurity OR "data breach" OR ransomware OR hacking OR malware OR "cyber attack")',
    "europe_politics": '("European Union" OR EU OR Brussels OR "European Commission" OR "European Parliament" OR Brexit OR Germany OR France OR Macron OR Scholz) AND (politics OR government OR election OR parliament OR diplomacy OR summit)',
    "middle_east_politics": '("Middle East" OR Israel OR Iran OR Gaza OR Syria OR Iraq OR Lebanon OR "Saudi Arabia" OR Yemen OR Palestinian) AND (politics OR government OR conflict OR diplomacy OR ceasefire OR negotiations OR sanctions)'
}

LANG = {"hl": "en-US", "gl": "US", "ceid": "US:en"}  # для русских: hl=ru, gl=RU, ceid=RU:ru

WINDOW_DAYS = 7          # стартовый размер окна
RSS_LIMIT = 100          # Google отдаёт не больше ~100 элементов на запрос
DELAY = (1.5, 3.0)       # случайная пауза между запросами, сек
MAX_RETRIES = 5
INCLUDE_TOPIC = True   # True -> добавить 4-й столбец с темой

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# ----------------------------- ЛОГИКА --------------------------------------


def build_url(query: str, d_from: date, d_to: date) -> str:
    """d_from включительно, d_to не включительно (так работает before:)."""
    q = f"{query} after:{d_from.isoformat()} before:{d_to.isoformat()}"
    params = "&".join(f"{k}={v}" for k, v in LANG.items())
    return f"https://news.google.com/rss/search?q={quote_plus(q)}&{params}"


def fetch(url: str) -> bytes | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r.content
            if r.status_code in (429, 503):
                wait = 30 * attempt + random.uniform(0, 10)
                print(f"  [{r.status_code}] лимит запросов, жду {wait:.0f} c...")
                time.sleep(wait)
                continue
            print(f"  [HTTP {r.status_code}] пропуск")
            return None
        except requests.RequestException as e:
            wait = 5 * attempt
            print(f"  [сеть] {e}; повтор через {wait} c")
            time.sleep(wait)
    return None


def parse_rss(content: bytes):
    """Возвращает список (datetime_utc, title, link)."""
    items = []
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return items
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        link = (it.findtext("link") or "").strip()
        pub = it.findtext("pubDate")
        if not (title and link and pub):
            continue
        try:
            dt = parsedate_to_datetime(pub).astimezone(timezone.utc)
        except (TypeError, ValueError):
            continue
        items.append((dt, title, link))
    return items


def collect(query: str, d_from: date, d_to: date):
    """Собирает новости за [d_from, d_to). Делит окно, если упёрлись в лимит."""
    content = fetch(build_url(query, d_from, d_to))
    time.sleep(random.uniform(*DELAY))
    if content is None:
        return []
    items = parse_rss(content)
    span = (d_to - d_from).days
    if len(items) >= RSS_LIMIT and span > 1:
        mid = d_from + timedelta(days=span // 2)
        return collect(query, d_from, mid) + collect(query, mid, d_to)
    return items


def load_progress(path: str) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_progress(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def load_seen(csv_path: str) -> set:
    seen = set()
    if os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 3:
                    seen.add(row[2])
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--out", default="news.csv")
    ap.add_argument("--window", type=int, default=WINDOW_DAYS)
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) + timedelta(days=1)  # включая сегодняшний день
    progress_path = args.out + ".progress.json"
    progress = load_progress(progress_path)  # {topic: "YYYY-MM-DD" (следующая дата)}
    seen = load_seen(args.out)

    new_file = not os.path.exists(args.out)
    with open(args.out, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=";", quoting=csv.QUOTE_ALL)
        if new_file:
            header = ["date", "news", "link"] + (["topic"] if INCLUDE_TOPIC else [])
            writer.writerow(header)

        for topic, query in TOPICS.items():
            cur = date.fromisoformat(progress.get(topic, start.isoformat()))
            cur = max(cur, start)
            print(f"\n=== Тема: {topic} (с {cur}) ===")
            while cur < end:
                nxt = min(cur + timedelta(days=args.window), end)
                items = collect(query, cur, nxt)
                added = 0
                for dt, title, link in sorted(items, key=lambda x: x[0]):
                    if link in seen:
                        continue
                    seen.add(link)
                    row = [dt.strftime("%Y-%m-%d %H:%M"), title, link]
                    if INCLUDE_TOPIC:
                        row.append(topic)
                    writer.writerow(row)
                    added += 1
                f.flush()
                print(f"{cur} -> {nxt}: найдено {len(items)}, новых {added}")
                cur = nxt
                progress[topic] = cur.isoformat()
                save_progress(progress_path, progress)

    print(f"\nГотово. Файл: {args.out}")


if __name__ == "__main__":
    main()