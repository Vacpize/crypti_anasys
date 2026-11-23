#!/usr/bin/env python3
"""
Fine-tunes an OpenAI model (default gpt-4.1-mini) to analyse a news headline on the fly.

Input:  one headline.
Output: JSON with the same fields as news_bd*_annotated.csv:
        globality, clickbait, actors, subjects, sentiment, sentiment_score,
        crypto_impact, gold_impact, economy_impact

Training data comes from news_bd1_annotated.csv / news_bd2_annotated.csv.
Split is by date, so the test set is "the future" for the model:
  train  - older news, validation - next slice, test - the newest news (never sent for training).

Install:  python3 -m pip install --user --break-system-packages openai pandas
Key:      OPEN_AI_API_KEY=sk-... (or OPENAI_API_KEY) in the .env file

Commands:
  python3 finetune_openai.py prepare                  # writes finetune_data/*.jsonl, prints token/size estimate
  python3 finetune_openai.py validate                 # local copy of OpenAI's file checks (+ free moderation check)
  python3 finetune_openai.py start                    # uploads files and starts the job (PAID)
  python3 finetune_openai.py status                   # job status + last events
  python3 finetune_openai.py eval --n 200             # compares the tuned model with the labels on the test set (PAID, small)
  python3 finetune_openai.py ask "Fed cuts rates by 50 bps"   # analyse one headline with the tuned model

Options for prepare: --train 5000 --val 500 --test 1000 (how many headlines go into each file)
Options for start:   --model gpt-4.1-mini-2025-04-14 --epochs auto --suffix news-analyst
"""

import argparse
import json
import os
import sys
import time

import pandas as pd
from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "finetune_data")
STATE = os.path.join(DATA_DIR, "job.json")
INPUTS = [os.path.join(HERE, "news_bd1_annotated.csv"), os.path.join(HERE, "news_bd2_annotated.csv")]
DEFAULT_MODEL = "gpt-4.1-mini-2025-04-14"   # gpt-6.x / gpt-5.x do not support fine-tuning

SYSTEM = (
    "You analyse one news headline for a study of how news moves Bitcoin, Ethereum and gold. "
    "Answer with JSON only: globality 0..1 (how much the event concerns global change), "
    "clickbait 0..1 (how emotional / sensational the wording is), actors (countries, officials, "
    "companies, organisations, cryptocurrencies; never the outlet name), subjects (1-3 topic codes), "
    "sentiment positive/negative/neutral and sentiment_score -1..1 (is the event good or bad in general), "
    "crypto_impact, gold_impact, economy_impact -1..1 (expected market effect, 0 = none)."
)
FIELDS = ["globality", "clickbait", "actors", "subjects", "sentiment", "sentiment_score",
          "crypto_impact", "gold_impact", "economy_impact"]
NUM_FIELDS = ["globality", "clickbait", "sentiment_score", "crypto_impact", "gold_impact", "economy_impact"]


# ----------------------------- HELPERS -------------------------------------


def load_env():
    """Reads KEY=VALUE lines from .env; accepts OPEN_AI_API_KEY as an alias of OPENAI_API_KEY."""
    path = os.path.join(HERE, ".env")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    v = v.strip().strip('"').strip("'")
                    if v:
                        os.environ.setdefault(k.strip(), v)
    if not os.environ.get("OPENAI_API_KEY") and os.environ.get("OPEN_AI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["OPEN_AI_API_KEY"]
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("No OpenAI key: put OPENAI_API_KEY=sk-... into the .env file.")


def load_news():
    df = pd.concat([pd.read_csv(p, sep=";", index_col=0, encoding="utf-8-sig") for p in INPUTS])
    missing = [c for c in FIELDS if c not in df]
    if missing:
        sys.exit(f"Columns {missing} are missing - run annotate_news.py (both passes) first.")
    df = df.dropna(subset=["globality", "crypto_impact"])
    df["time"] = pd.to_datetime(df["date"], format="mixed")
    df = df.drop_duplicates(subset="news").sort_values("time").reset_index(drop=True)
    return df


def label(row):
    """Target answer for one headline, in a fixed key order."""
    split = lambda s: [x for x in str(s).split("|") if x and x != "nan"]
    out = {}
    for c in FIELDS:
        v = row[c]
        if c in ("actors", "subjects"):
            out[c] = split(v)
        elif c == "sentiment":
            out[c] = v
        else:
            out[c] = round(float(v), 2)
    return out


def example(row):
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": row["news"]},
        {"role": "assistant", "content": json.dumps(label(row), ensure_ascii=False)},
    ]}


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for _, r in rows.iterrows():
            f.write(json.dumps(example(r), ensure_ascii=False) + "\n")


def load_state():
    if os.path.exists(STATE):
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(state):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(STATE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def parse_answer(text):
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


# ----------------------------- COMMANDS ------------------------------------


def cmd_prepare(args):
    df = load_news()
    n_test = args.test
    test = df.tail(n_test)                              # the newest news
    rest = df.iloc[: len(df) - n_test]
    val_pool = rest.tail(max(args.val * 5, args.val))   # slice right before the test period
    train_pool = rest.iloc[: len(rest) - len(val_pool)]

    # Balanced-ish sample: keep every headline with a strong market signal, fill the rest at random,
    # otherwise ~60% of examples would teach the model to answer "impact 0".
    strong = train_pool[(train_pool[["crypto_impact", "gold_impact", "economy_impact"]].abs() >= 0.2).any(axis=1)]
    strong = strong.sample(min(len(strong), args.train // 2), random_state=1)
    filler = train_pool.drop(strong.index).sample(args.train - len(strong), random_state=1)
    train = pd.concat([strong, filler]).sample(frac=1, random_state=1)
    val = val_pool.sample(min(args.val, len(val_pool)), random_state=1)

    os.makedirs(DATA_DIR, exist_ok=True)
    write_jsonl(os.path.join(DATA_DIR, "train.jsonl"), train)
    write_jsonl(os.path.join(DATA_DIR, "val.jsonl"), val)
    write_jsonl(os.path.join(DATA_DIR, "test.jsonl"), test)

    # rough size: ~4 characters per token
    chars = sum(len(json.dumps(example(r), ensure_ascii=False)) for _, r in train.iterrows())
    tokens = chars / 4
    print(f"train: {len(train):,} ({len(strong):,} with a strong market signal), "
          f"{train['time'].min():%Y-%m-%d} .. {train['time'].max():%Y-%m-%d}")
    print(f"val:   {len(val):,}, {val['time'].min():%Y-%m-%d} .. {val['time'].max():%Y-%m-%d}")
    print(f"test:  {len(test):,}, {test['time'].min():%Y-%m-%d} .. {test['time'].max():%Y-%m-%d} (not uploaded)")
    print(f"~{tokens / 1e6:.2f}M tokens per epoch; billed training tokens = that x number of epochs")
    print(f"files in {DATA_DIR}")


ALLOWED_ROLES = {"system", "developer", "user", "assistant", "tool", "function"}
MESSAGE_KEYS = {"role", "content", "name", "weight", "tool_calls", "function_call", "tool_call_id"}
MAX_EXAMPLE_TOKENS = 65_536      # training context limit for gpt-4.1-mini
MAX_FILE_BYTES = 512 * 1024 ** 2


def check_file(path, enc):
    """Format checks from OpenAI's fine-tuning data guide. Returns (errors, stats)."""
    errors, tokens, seen, dups = [], [], set(), 0
    raw = open(path, "rb").read()
    if raw.startswith(b"\xef\xbb\xbf"):
        errors.append("file starts with a UTF-8 BOM")
    if len(raw) > MAX_FILE_BYTES:
        errors.append(f"file is {len(raw) / 1024 ** 2:.0f} MB, limit is 512 MB")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        return [f"not valid UTF-8: {e}"], {}
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    for i, line in enumerate(lines, 1):
        where = f"line {i}"
        if not line.strip():
            errors.append(f"{where}: empty line")
            continue
        try:
            ex = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"{where}: invalid JSON ({e})")
            continue
        if not isinstance(ex, dict) or set(ex) != {"messages"}:
            errors.append(f"{where}: top level must be exactly {{'messages': [...]}}, got keys {list(ex)[:5]}")
            continue
        msgs = ex["messages"]
        if not isinstance(msgs, list) or not msgs:
            errors.append(f"{where}: 'messages' must be a non-empty list")
            continue
        n_tok = 3
        for j, m in enumerate(msgs):
            if not isinstance(m, dict):
                errors.append(f"{where} msg {j}: not an object")
                continue
            if set(m) - MESSAGE_KEYS:
                errors.append(f"{where} msg {j}: unknown keys {sorted(set(m) - MESSAGE_KEYS)}")
            if m.get("role") not in ALLOWED_ROLES:
                errors.append(f"{where} msg {j}: bad role {m.get('role')!r}")
            c = m.get("content")
            if m.get("role") != "assistant" or not m.get("tool_calls"):
                if not isinstance(c, str) or not c.strip():
                    errors.append(f"{where} msg {j}: content must be a non-empty string")
                    continue
            if isinstance(c, str):
                if c != c.strip():
                    errors.append(f"{where} msg {j}: leading/trailing whitespace")
                if "\x00" in c:
                    errors.append(f"{where} msg {j}: NUL character")
                n_tok += 4 + len(enc.encode(c))
        roles = [m.get("role") for m in msgs if isinstance(m, dict)]
        if "assistant" not in roles:
            errors.append(f"{where}: no assistant message")
        elif roles[-1] != "assistant":
            errors.append(f"{where}: last message is not from the assistant")
        if n_tok > MAX_EXAMPLE_TOKENS:
            errors.append(f"{where}: {n_tok} tokens, over the {MAX_EXAMPLE_TOKENS} limit")
        tokens.append(n_tok)
        key = line.strip()
        dups += key in seen
        seen.add(key)
    if len(tokens) < 10:
        errors.append(f"only {len(tokens)} examples, OpenAI needs at least 10")
    stats = {"examples": len(tokens), "tokens": sum(tokens), "max_tokens": max(tokens, default=0),
             "duplicates": dups, "bytes": len(raw)}
    return errors, stats


def moderation_check(client, path, batch=50):
    """Runs every example through the free moderation endpoint, like OpenAI's safety screening."""
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            msgs = json.loads(line)["messages"]
            texts.append("\n".join(m["content"] for m in msgs if m["role"] != "system"))
    flagged = []
    for i in range(0, len(texts), batch):
        res = client.moderations.create(model="omni-moderation-latest", input=texts[i:i + batch])
        for j, r in enumerate(res.results):
            if r.flagged:
                cats = [k for k, v in r.categories.model_dump().items() if v]
                flagged.append((i + j + 1, cats, texts[i + j].split("\n")[0]))
    return len(texts), flagged


def cmd_validate(args):
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")
    ok = True
    for name in ("train", "val"):
        path = os.path.join(DATA_DIR, f"{name}.jsonl")
        errors, st = check_file(path, enc)
        print(f"{name}.jsonl: {st.get('examples', 0):,} examples, {st.get('tokens', 0):,} tokens, "
              f"longest {st.get('max_tokens', 0)} tokens, {st.get('duplicates', 0)} duplicates, "
              f"{st.get('bytes', 0) / 1024:.0f} KB")
        for e in errors[:20]:
            print(f"  ERROR {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors")
        ok &= not errors
    if args.moderation:
        load_env()
        client = OpenAI()
        for name in ("train", "val"):
            n, flagged = moderation_check(client, os.path.join(DATA_DIR, f"{name}.jsonl"))
            print(f"{name}.jsonl moderation: {len(flagged)} of {n:,} examples flagged")
            for line_no, cats, headline in flagged[:15]:
                print(f"  line {line_no} {cats}: {headline[:100]}")
            if len(flagged) > 15:
                print(f"  ... and {len(flagged) - 15} more")
    print("\nFormat: OK" if ok else "\nFormat: problems found")


def cmd_start(args):
    client = OpenAI()
    train_path = os.path.join(DATA_DIR, "train.jsonl")
    val_path = os.path.join(DATA_DIR, "val.jsonl")
    if not os.path.exists(train_path):
        sys.exit("Run prepare first.")
    print("Uploading files ...")
    with open(train_path, "rb") as f:
        train_file = client.files.create(file=f, purpose="fine-tune")
    with open(val_path, "rb") as f:
        val_file = client.files.create(file=f, purpose="fine-tune")
    epochs = args.epochs if args.epochs == "auto" else int(args.epochs)
    job = client.fine_tuning.jobs.create(
        model=args.model,
        training_file=train_file.id,
        validation_file=val_file.id,
        suffix=args.suffix,
        seed=42,
        method={"type": "supervised",
                "supervised": {"hyperparameters": {"n_epochs": epochs}}},
    )
    state = load_state()
    state.setdefault("jobs", []).append({"job_id": job.id, "model": args.model,
                                         "training_file": train_file.id, "validation_file": val_file.id})
    save_state(state)
    print(f"Job {job.id} created for {args.model}: {job.status}")
    print("Check progress with: python3 finetune_openai.py status")


def load_jobs(state):
    """All tracked jobs; also reads the old single-job state format."""
    jobs = list(state.get("jobs", []))
    if "job_id" in state:   # old format: one job at the top level
        jobs.insert(0, {k: state[k] for k in ("job_id", "model") if k in state})
        state["jobs"] = jobs
        for k in ("job_id", "model", "training_file", "validation_file"):
            state.pop(k, None)
    return jobs


def cmd_status(args):
    client = OpenAI()
    state = load_state()
    jobs = load_jobs(state)
    if not jobs:
        sys.exit("No job yet - run start first.")
    for entry in jobs:
        job = client.fine_tuning.jobs.retrieve(entry["job_id"])
        print(f"{job.id}: {job.status}  base={job.model}  tuned={job.fine_tuned_model}")
        if job.trained_tokens:
            print(f"  trained tokens: {job.trained_tokens:,}")
        if job.error and job.error.message:
            print(f"  error: {job.error.message}")
        events = client.fine_tuning.jobs.list_events(job.id, limit=args.events)
        for e in reversed(events.data):
            print(f"  {time.strftime('%Y-%m-%d %H:%M', time.localtime(e.created_at))}  {e.message}")
        entry["status"] = job.status
        if job.fine_tuned_model:
            entry["tuned_model"] = job.fine_tuned_model
            state.setdefault("tuned_model", job.fine_tuned_model)   # first finished model is the default
    save_state(state)


def tuned_model():
    model = MODEL_OVERRIDE or load_state().get("tuned_model")
    if not model:
        sys.exit("The tuned model is not ready yet - run status until it shows tuned=ft:...")
    return model


MODEL_OVERRIDE = None   # set by --model on eval / ask


def analyse(client, model, headline):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": headline}],
    )
    return resp.choices[0].message.content


def cmd_ask(args):
    client = OpenAI()
    text = analyse(client, tuned_model(), args.headline)
    data = parse_answer(text)
    print(json.dumps(data, ensure_ascii=False, indent=2) if data else text)


def cmd_eval(args):
    client = OpenAI()
    model = tuned_model()
    rows = []
    with open(os.path.join(DATA_DIR, "test.jsonl"), encoding="utf-8") as f:
        for line in f:
            msgs = json.loads(line)["messages"]
            rows.append((msgs[1]["content"], json.loads(msgs[2]["content"])))
    rows = rows[-args.n:]
    err = {c: [] for c in NUM_FIELDS}
    sent_ok = subj_ok = bad_json = 0
    for i, (headline, truth) in enumerate(rows, 1):
        pred = parse_answer(analyse(client, model, headline))
        if not pred:
            bad_json += 1
            continue
        for c in NUM_FIELDS:
            try:
                err[c].append(abs(float(pred.get(c)) - truth[c]))
            except (TypeError, ValueError):
                pass
        sent_ok += pred.get("sentiment") == truth["sentiment"]
        subj_ok += bool(pred.get("subjects")) and bool(truth["subjects"]) \
            and pred["subjects"][0] == truth["subjects"][0]
        if i % 50 == 0:
            print(f"  {i}/{len(rows)}")
    ok = len(rows) - bad_json
    print(f"\n{model} on {len(rows)} newest headlines (labels from annotate_news.py):")
    print(f"  valid JSON:            {ok}/{len(rows)}")
    print(f"  sentiment accuracy:    {sent_ok / max(ok, 1):.1%}")
    print(f"  main subject accuracy: {subj_ok / max(ok, 1):.1%}")
    for c in NUM_FIELDS:
        if err[c]:
            print(f"  {c:<16} mean abs error {sum(err[c]) / len(err[c]):.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("prepare")
    s.add_argument("--train", type=int, default=5000)
    s.add_argument("--val", type=int, default=500)
    s.add_argument("--test", type=int, default=1000)
    s = sub.add_parser("validate")
    s.add_argument("--no-moderation", dest="moderation", action="store_false",
                   help="skip the moderation check (it needs the API key, but is free)")
    s = sub.add_parser("start")
    s.add_argument("--model", default=DEFAULT_MODEL)
    s.add_argument("--epochs", default="auto")
    s.add_argument("--suffix", default="news-analyst")
    s = sub.add_parser("status")
    s.add_argument("--events", type=int, default=10)
    s = sub.add_parser("eval")
    s.add_argument("--n", type=int, default=200)
    s.add_argument("--model", help="tuned model id (ft:...); default: the first finished job")
    s = sub.add_parser("ask")
    s.add_argument("headline")
    s.add_argument("--model", help="tuned model id (ft:...); default: the first finished job")
    args = ap.parse_args()
    global MODEL_OVERRIDE
    if args.cmd in ("eval", "ask"):
        MODEL_OVERRIDE = args.model

    if args.cmd not in ("prepare", "validate"):
        load_env()
    {"prepare": cmd_prepare, "validate": cmd_validate, "start": cmd_start, "status": cmd_status,
     "eval": cmd_eval, "ask": cmd_ask}[args.cmd](args)


if __name__ == "__main__":
    main()
