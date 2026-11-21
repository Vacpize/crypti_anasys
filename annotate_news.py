#!/usr/bin/env python3
"""
Annotates news headlines with Claude (Haiku 5.5) and writes *_annotated.csv files.

New columns:
  globality        0..1   how much the event matters for global markets / the world
  clickbait        0..1   how emotional / clickbait the headline is
  actors           "|"-separated: countries, politicians, officials, companies, organisations
  subjects         "|"-separated codes from SUBJECTS (inflation, crypto_market, ...)
  sentiment        positive / negative / neutral
  sentiment_score  -1..1  (negative ... positive)

Install:  uv venv .venv && uv pip install --python .venv/bin/python anthropic pandas
Key:      put ANTHROPIC_API_KEY=sk-ant-... into the .env file (it is in .gitignore)

Second pass (expected market effect, -1..1): add --task impact before the command, e.g.
  python annotate_news.py --task impact submit
  crypto_impact / gold_impact / economy_impact  bearish(-1) ... bullish(+1) for crypto, gold, global economy

Commands:
  python annotate_news.py sample --n 20     # quick synchronous test, prints results
  python annotate_news.py submit            # sends everything via Batch API (50% cheaper)
  python annotate_news.py status            # shows batch progress
  python annotate_news.py collect           # downloads results, writes *_annotated.csv
  python annotate_news.py submit            # run again after collect: re-sends only failed chunks

Resumable: results are cached in annotation_work/results.jsonl, finished chunks are never re-sent.
"""

import argparse
import csv
import json
import os
import random
import sys

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

# ----------------------------- SETTINGS ------------------------------------

MODEL = "claude-haiku-5-5"
CHUNK = 40                     # headlines per request
MAX_TOKENS = 8000              # Haiku 5.5 thinks by default; thinking counts toward max_tokens
EFFORT = "low"
INPUTS = ["news_bd1.csv", "news_bd2.csv"]
WORK_DIR = "annotation_work"
RESULTS = os.path.join(WORK_DIR, "results.jsonl")   # one line per finished chunk
STATE = os.path.join(WORK_DIR, "batches.json")      # submitted batch ids

SUBJECTS = [
    "inflation", "interest_rates", "central_bank_policy", "currency_exchange",
    "stock_market", "crypto_market", "crypto_regulation", "crypto_adoption",
    "gold_commodities", "energy_oil", "banking_finance", "taxes_fiscal",
    "trade_tariffs", "sanctions", "war_conflict", "geopolitics_diplomacy",
    "elections_politics", "regulation_law", "economy_growth", "labor_employment",
    "real_estate", "corporate_business", "ipo_mergers", "technology_ai",
    "semiconductors", "cybersecurity", "weather_disaster", "other",
]
SENTIMENTS = ["positive", "negative", "neutral"]

SYSTEM = f"""You annotate news headlines for a dataset used to study how news moves Bitcoin, Ethereum and gold prices.

For every headline return one item with:
- id: the id given in the input.
- globality: number 0..1 — how much the event concerns global change (world economy, global markets, many countries).
  0.0 = local/niche (a single small company, a city event, a how-to article, a product review)
  0.3 = national or sector-level significance
  0.6 = major economy / large market / big multinational / regional conflict
  1.0 = world-scale shock (global crisis, major war, Fed rate decision, global pandemic, BTC ETF approval)
- clickbait: number 0..1 — how emotional / sensational / clickbait the wording is (not the event itself).
  0.0 = neutral factual wording; 0.5 = noticeable hype or loaded words; 1.0 = pure clickbait ("You won't believe...", "SHOCKING", "to the moon", heavy exaggeration).
- actors: list of acting figures explicitly named or clearly implied: countries, governments, politicians and officials, central banks, companies, organisations, notable people, specific cryptocurrencies. Use short canonical English names ("United States", "Federal Reserve", "Donald Trump", "Binance", "Bitcoin"). Do NOT include the news outlet name that follows the final " - ". Empty list if none.
- subjects: 1-3 codes from this list, most relevant first: {", ".join(SUBJECTS)}.
- sentiment: positive / negative / neutral — the character of the news (good or bad development), not the wording.
- sentiment_score: number -1..1 (-1 very negative, 0 neutral, 1 very positive), consistent with sentiment.

Return exactly one item per input id, in the same order."""

SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "globality": {"type": "number"},
                    "clickbait": {"type": "number"},
                    "actors": {"type": "array", "items": {"type": "string"}},
                    "subjects": {"type": "array", "items": {"type": "string", "enum": SUBJECTS}},
                    "sentiment": {"type": "string", "enum": SENTIMENTS},
                    "sentiment_score": {"type": "number"},
                },
                "required": ["id", "globality", "clickbait", "actors", "subjects",
                             "sentiment", "sentiment_score"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}

NEW_COLS = ["globality", "clickbait", "actors", "subjects", "sentiment", "sentiment_score"]

# Second pass ("impact" task): expected market effect, unlike sentiment (general good/bad of the event).
IMPACT_SYSTEM = """You annotate news headlines for a dataset used to study how news moves Bitcoin, Ethereum and gold prices.

For every headline estimate the expected direction and strength of its effect on three things over the following days, as a market analyst would judge it at the moment of publication. Numbers are -1..1:
  -1 = strongly bearish / harmful, 0 = no meaningful effect, +1 = strongly bullish / beneficial.
Strength must reflect both direction and how much the event matters for that market: most headlines (local news, how-to articles, product reviews, single small companies) should be 0 or close to 0.

- id: the id given in the input.
- crypto_impact: effect on cryptocurrency prices (Bitcoin, Ethereum, the crypto market as a whole).
  Examples: spot Bitcoin ETF approved +0.8; major exchange hacked or collapses -0.8; country bans crypto -0.6; Fed cuts rates / more liquidity +0.4; Fed hikes rates -0.4; a small altcoin partnership +0.1.
- gold_impact: effect on the gold price. Gold is a safe haven: war, sanctions, financial crises, high inflation, rate cuts and a weak dollar are positive for gold; rate hikes, a strong dollar and calm risk-on markets are negative.
- economy_impact: effect on the global economy and stock markets (growth, corporate earnings, risk appetite).

Judge the market effect, not whether the event is morally good or bad: a war is negative for the economy but positive for gold. Return exactly one item per input id, in the same order."""

IMPACT_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "crypto_impact": {"type": "number"},
                    "gold_impact": {"type": "number"},
                    "economy_impact": {"type": "number"},
                },
                "required": ["id", "crypto_impact", "gold_impact", "economy_impact"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}
IMPACT_COLS = ["crypto_impact", "gold_impact", "economy_impact"]

TASKS = {
    "base": (SYSTEM, SCHEMA, RESULTS, STATE),
    "impact": (IMPACT_SYSTEM, IMPACT_SCHEMA,
               os.path.join(WORK_DIR, "results_impact.jsonl"),
               os.path.join(WORK_DIR, "batches_impact.json")),
}


def use_task(name):
    """Switches prompt, schema and work files to the chosen annotation pass."""
    global SYSTEM, SCHEMA, RESULTS, STATE
    SYSTEM, SCHEMA, RESULTS, STATE = TASKS[name]

# ----------------------------- DATA ----------------------------------------


def load_rows():
    """Returns {file: (header, rows)}; rows keep the original columns."""
    data = {}
    for path in INPUTS:
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f, delimiter=";")
            header = next(reader)
            data[path] = (header, [r for r in reader if len(r) >= 5])
    return data


def all_headlines(data):
    """[(row_id, headline)] over all files; row_id = first CSV column (unique across files)."""
    out = []
    for _, rows in data.values():
        out.extend((int(r[0]), r[2]) for r in rows)
    return out


def chunks(items):
    return {f"c{i // CHUNK:05d}": items[i:i + CHUNK] for i in range(0, len(items), CHUNK)}


def build_params(chunk):
    lines = "\n".join(json.dumps({"id": rid, "headline": text}, ensure_ascii=False)
                      for rid, text in chunk)
    return MessageCreateParamsNonStreaming(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": f"Headlines (JSON lines):\n{lines}"}],
    )


def parse_message(msg, chunk):
    """Returns {row_id: annotation} or None if the answer is unusable."""
    if msg.stop_reason not in ("end_turn", "stop_sequence"):
        return None
    text = next((b.text for b in msg.content if b.type == "text"), "")
    try:
        items = json.loads(text)["items"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    wanted = {rid for rid, _ in chunk}
    got = {}
    for it in items:
        if it.get("id") in wanted:
            got[it["id"]] = it
    if len(got) < len(wanted) * 0.9:   # model skipped too many -> retry the chunk
        return None
    return got


def load_done(path=None):
    path = path or RESULTS
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                done[rec["chunk"]] = rec["items"]
    return done


def save_chunk(cid, got):
    os.makedirs(WORK_DIR, exist_ok=True)
    with open(RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps({"chunk": cid, "items": {str(k): v for k, v in got.items()}},
                           ensure_ascii=False) + "\n")


def load_state():
    if os.path.exists(STATE):
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    return {"batches": []}


def save_state(state):
    os.makedirs(WORK_DIR, exist_ok=True)
    with open(STATE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


# ----------------------------- COMMANDS ------------------------------------


def cmd_sample(client, args):
    data = load_rows()
    items = all_headlines(data)
    random.seed(args.seed)
    chunk = random.sample(items, args.n)
    msg = client.messages.create(**build_params(chunk))
    got = parse_message(msg, chunk)
    print(f"usage: in={msg.usage.input_tokens} out={msg.usage.output_tokens} "
          f"cache_read={msg.usage.cache_read_input_tokens} stop={msg.stop_reason}")
    if got is None:
        print("Could not parse the answer:")
        print(msg.content)
        return
    for rid, text in chunk:
        a = got.get(rid)
        print(f"\n{text}")
        if a:
            print("  " + "  ".join(f"{k}={v}" for k, v in a.items() if k != "id"))
        else:
            print("  (missing)")


def cmd_submit(client, args):
    data = load_rows()
    all_chunks = chunks(all_headlines(data))
    done = load_done()
    state = load_state()
    pending = set()   # chunks sitting in batches that are not collected yet
    for b in state["batches"]:
        if not b.get("collected"):
            pending.update(b["chunks"])
    todo = [cid for cid in all_chunks if cid not in done and cid not in pending]
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        print("Nothing to submit: everything is done or sits in a batch that is not collected yet.")
        return
    print(f"Submitting {len(todo)} requests ({len(todo) * CHUNK} headlines max) to {MODEL} ...")
    for start in range(0, len(todo), 10_000):   # keep each batch well under the 256 MB limit
        part = todo[start:start + 10_000]
        batch = client.messages.batches.create(requests=[
            Request(custom_id=cid, params=build_params(all_chunks[cid])) for cid in part
        ])
        state["batches"].append({"id": batch.id, "chunks": part, "collected": False})
        save_state(state)
        print(f"  batch {batch.id}: {len(part)} requests")
    print("Done. Check progress with: python annotate_news.py status")


def cmd_status(client, args):
    for b in load_state()["batches"]:
        if b.get("collected"):
            print(f"{b['id']}: collected")
            continue
        info = client.messages.batches.retrieve(b["id"])
        c = info.request_counts
        print(f"{b['id']}: {info.processing_status}  processing={c.processing} "
              f"succeeded={c.succeeded} errored={c.errored} expired={c.expired}")


def cmd_collect(client, args):
    data = load_rows()
    all_chunks = chunks(all_headlines(data))
    done = load_done()
    state = load_state()
    for b in state["batches"]:
        if b.get("collected"):
            continue
        info = client.messages.batches.retrieve(b["id"])
        if info.processing_status != "ended":
            print(f"{b['id']} is still {info.processing_status}, skipping")
            continue
        ok = bad = 0
        for res in client.messages.batches.results(b["id"]):
            cid = res.custom_id
            if cid in done:
                continue
            got = parse_message(res.result.message, all_chunks[cid]) \
                if res.result.type == "succeeded" else None
            if got is None:
                bad += 1
                continue
            save_chunk(cid, got)
            done[cid] = {str(k): v for k, v in got.items()}
            ok += 1
        b["collected"] = True
        save_state(state)
        print(f"{b['id']}: saved {ok} chunks, failed {bad}")
    write_outputs(data)
    missing = [cid for cid in all_chunks if cid not in done]
    if missing:
        print(f"{len(missing)} chunks still missing -> run submit again, then collect")


def clamp(x, lo, hi):
    try:
        return round(min(max(float(x), lo), hi), 3)
    except (TypeError, ValueError):
        return ""


def write_outputs(data):
    """Rewrites *_annotated.csv from everything collected so far (both passes)."""
    ann, imp = {}, {}
    for target, task in ((ann, "base"), (imp, "impact")):
        for items in load_done(TASKS[task][2]).values():
            for k, v in items.items():
                target[int(k)] = v
    for path, (header, rows) in data.items():
        out = path.replace(".csv", "_annotated.csv")
        filled = 0
        with open(out, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(header + NEW_COLS + (IMPACT_COLS if imp else []))
            for r in rows:
                a = ann.get(int(r[0]))
                i = imp.get(int(r[0]))
                extra = ([clamp(i[c], -1, 1) for c in IMPACT_COLS] if i else [""] * len(IMPACT_COLS)) \
                    if imp else []
                if a:
                    filled += 1
                    w.writerow(r + [
                        clamp(a["globality"], 0, 1),
                        clamp(a["clickbait"], 0, 1),
                        "|".join(dict.fromkeys(s.strip() for s in a["actors"] if s.strip())),
                        "|".join(a["subjects"]),
                        a["sentiment"],
                        clamp(a["sentiment_score"], -1, 1),
                    ] + extra)
                else:
                    w.writerow(r + [""] * len(NEW_COLS) + extra)
        note = f", impact columns for {sum(1 for r in rows if int(r[0]) in imp)}" if imp else ""
        print(f"{out}: {filled}/{len(rows)} rows annotated{note}")


def load_env(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")):
    """Reads KEY=VALUE lines from .env into os.environ (existing variables win)."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                v = v.strip().strip('"').strip("'")
                if v:
                    os.environ.setdefault(k.strip(), v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=TASKS, default="base",
                    help="base = globality/clickbait/actors/subjects/sentiment; "
                         "impact = crypto_impact/gold_impact/economy_impact")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--n", type=int, default=20)
    s.add_argument("--seed", type=int, default=1)
    s = sub.add_parser("submit")
    s.add_argument("--limit", type=int, default=0, help="send only the first N chunks (for a trial run)")
    sub.add_parser("status")
    sub.add_parser("collect")
    args = ap.parse_args()

    use_task(args.task)
    load_env()
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        sys.exit("No API key: put ANTHROPIC_API_KEY=sk-ant-... into the .env file next to this script.")
    client = anthropic.Anthropic()
    {"sample": cmd_sample, "submit": cmd_submit, "status": cmd_status, "collect": cmd_collect}[args.cmd](client, args)


if __name__ == "__main__":
    main()
