#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adds a "description" column to the news CSV produced by news_parser.py.

Steps for every row:
  1. The Google News link (news.google.com/rss/articles/...) is decoded to the real article URL.
  2. The article page is downloaded and the description is taken from meta tags
     (og:description, twitter:description, <meta name="description">).
  3. Rows are written to a NEW file; the original CSV is not modified.

Install:  pip install -U requests googlenewsdecoder
Run:      python add_descriptions.py --in news.csv --out news_with_desc.csv
Test one link (shows exactly where it fails):
          python add_descriptions.py --test "https://news.google.com/rss/articles/...."
Debug a run (prints the reason for every failed row):
          python add_descriptions.py --in news.csv --out test.csv --debug

Resumable: run the same command again after a stop, finished links are skipped.
"""

import argparse
import base64
import csv
import html
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser

import requests

try:
    from googlenewsdecoder import gnewsdecoder
except ImportError:
    gnewsdecoder = None

# ----------------------------- SETTINGS ------------------------------------

WORKERS = 3              # parallel downloads (Google blocks the decoder if this is high)
BATCH = 40               # rows processed per batch
BATCH_DELAY = (1.0, 2.5) # pause between batches, sec
TIMEOUT = 20
MAX_DESC_LEN = 500       # truncate long descriptions
MAX_PAGE_BYTES = 300_000 # only the beginning of the page is needed (the <head>)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "cross-site",
    "Sec-Fetch-User": "?1",
}

DEBUG = False


def log(msg: str):
    if DEBUG:
        print("   [debug]", msg)


# ----------------------------- LOGIC ---------------------------------------


class MetaParser(HTMLParser):
    """Collects description-like meta tags."""

    def __init__(self):
        super().__init__()
        self.found = {}

    def handle_starttag(self, tag, attrs):
        if tag != "meta":
            return
        a = {k.lower(): (v or "") for k, v in attrs}
        key = (a.get("property") or a.get("name") or "").lower()
        if key in ("og:description", "twitter:description", "description"):
            content = a.get("content", "").strip()
            if content and key not in self.found:
                self.found[key] = content


def clean(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > MAX_DESC_LEN:
        text = text[: MAX_DESC_LEN - 1].rstrip() + "…"
    return text


def decode_old_format(link: str) -> str:
    """Old-style Google links keep the real URL inside the base64 id (no network needed)."""
    m = re.search(r"/articles/([^?/#]+)", link)
    if not m:
        return ""
    token = m.group(1)
    try:
        raw = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4))
    except Exception:
        return ""
    found = re.search(rb"https?://[\x21-\x7e]+", raw)
    if not found:
        return ""
    return found.group(0).decode("ascii", errors="ignore")


def decode_via_google(link: str) -> str:
    """Own decoder: asks Google's batchexecute endpoint for the real URL (used if the library fails)."""
    m = re.search(r"/articles/([^?/#]+)", link)
    if not m:
        return ""
    art_id = m.group(1)
    try:
        s = requests.Session()
        s.headers.update(HEADERS)
        s.cookies.set("CONSENT", "YES+cb", domain=".google.com")
        r = s.get(f"https://news.google.com/rss/articles/{art_id}", timeout=TIMEOUT)
        log(f"google article page: HTTP {r.status_code}")
        sg = re.search(r'data-n-a-sg="([^"]+)"', r.text)
        ts = re.search(r'data-n-a-ts="([^"]+)"', r.text)
        if not (sg and ts):
            log("signature/timestamp not found on the Google page (consent page, block or format change)")
            return ""
        inner = [
            "garturlreq",
            [["X", "X", ["X", "X"], None, None, 1, 1, "US:en", None, 1, None, None, None, None, None, 0, 1],
             "X", "X", 1, [1, 1, 1], 1, 1, None, 0, 0, None, 0],
            art_id, ts.group(1), sg.group(1),
        ]
        payload = [[["Fbv4je", json.dumps(inner), None, "generic"]]]
        time.sleep(random.uniform(0.5, 1.5))
        r2 = s.post(
            "https://news.google.com/_/DotsSplashUi/data/batchexecute",
            data={"f.req": json.dumps(payload)},
            headers={"Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"},
            timeout=TIMEOUT,
        )
        log(f"google batchexecute: HTTP {r2.status_code}")
        if r2.status_code != 200:
            return ""
        data = json.loads(r2.text.split("\n\n")[1])[:-2]
        return json.loads(data[0][2])[1]
    except Exception as e:
        log(f"own decoder error: {e!r}")
        return ""


def resolve_url(link: str) -> str:
    """Turns a Google News redirect link into the real article URL."""
    if "news.google.com" not in link:
        return link

    url = decode_old_format(link)
    if url:
        log(f"decoded offline -> {url}")
        return url

    if gnewsdecoder is None:
        log("googlenewsdecoder is NOT installed (pip install -U googlenewsdecoder)")
    else:
        try:
            res = gnewsdecoder(link)
            if isinstance(res, dict) and (res.get("status") or res.get("success")) and res.get("decoded_url"):
                log(f"decoded by googlenewsdecoder -> {res.get('decoded_url')}")
                return res["decoded_url"]
            log(f"googlenewsdecoder failed, full answer: {res!r}")
        except Exception as e:
            log(f"googlenewsdecoder error: {e!r}")

    url = decode_via_google(link)
    if url:
        log(f"decoded by own decoder -> {url}")
    return url


def get_description(link: str) -> str:
    try:
        url = resolve_url(link)
        if not url:
            return ""
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT, stream=True)
        if r.status_code != 200:
            log(f"page returned HTTP {r.status_code}: {url}")
            return ""
        ctype = r.headers.get("Content-Type", "").lower()
        if "html" not in ctype:
            log(f"not an HTML page ({ctype}): {url}")
            return ""
        raw = next(r.iter_content(MAX_PAGE_BYTES), b"")
        r.close()
        text = raw.decode(r.encoding or "utf-8", errors="replace")
        parser = MetaParser()
        parser.feed(text)
        for key in ("og:description", "twitter:description", "description"):
            if key in parser.found:
                return clean(parser.found[key])
        log(f"no description meta tag on the page: {url}")
    except Exception as e:
        log(f"error: {e!r}")
    return ""


def load_done(path: str) -> set:
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f, delimiter=";")
            next(reader, None)
            for row in reader:
                if len(row) >= 3:
                    done.add(row[2])
    return done


def main():
    global DEBUG
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="news.csv")
    ap.add_argument("--out", default="news_with_desc.csv")
    ap.add_argument("--from-date", default="2017-08-17 07:00",
                    help='only process news from this moment: "YYYY-MM-DD" or "YYYY-MM-DD HH:MM" '
                         '(UTC, same as the CSV); default "2017-08-17 07:00"')
    ap.add_argument("--debug", action="store_true", help="print the reason for every failed row")
    ap.add_argument("--test", metavar="LINK", help="test a single link and exit")
    args = ap.parse_args()
    DEBUG = args.debug

    if args.test:
        DEBUG = True
        print("googlenewsdecoder installed:", gnewsdecoder is not None)
        print("RESULT:", get_description(args.test) or "(empty)")
        return

    if gnewsdecoder is None:
        print("WARNING: googlenewsdecoder is not installed (pip install -U googlenewsdecoder).")
        print("New-style Google links cannot be resolved without it.\n")

    done = load_done(args.out)
    print(f"Already processed: {len(done)}")

    with open(args.inp, encoding="utf-8-sig", newline="") as fin:
        reader = csv.reader(fin, delimiter=";")
        header = next(reader)
        rows = []
        skipped_old = 0
        for r in reader:
            if len(r) < 3 or r[2] in done:
                continue
            if r[0] < args.from_date:  # "YYYY-MM-DD HH:MM" strings compare correctly
                skipped_old += 1
                continue
            rows.append(r)

    print(f"Skipped (older than {args.from_date}): {skipped_old}")
    print(f"Left to process: {len(rows)}")

    new_file = not os.path.exists(args.out)
    with open(args.out, "a", encoding="utf-8-sig", newline="") as fout:
        writer = csv.writer(fout, delimiter=";", quoting=csv.QUOTE_ALL)
        if new_file:
            writer.writerow(header + ["description"])

        total = len(rows)
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for i in range(0, total, BATCH):
                batch = rows[i : i + BATCH]
                descs = list(pool.map(lambda r: get_description(r[2]), batch))
                for r, d in zip(batch, descs):
                    writer.writerow(r + [d])
                fout.flush()
                filled = sum(1 for d in descs if d)
                print(f"{min(i + BATCH, total)}/{total} done, descriptions found in this batch: {filled}/{len(batch)}")
                if i == 0 and filled == 0:
                    print("   First batch is empty. Stop (Ctrl+C) and run with --test \"<link>\" to see why.")
                time.sleep(random.uniform(*BATCH_DELAY))

    print(f"\nDone. File: {args.out}")


if __name__ == "__main__":
    main()