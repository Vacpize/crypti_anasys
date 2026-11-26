#!/usr/bin/env bash
# Sequential queue of runs on DGX Spark (one GPU job at a time). Safe to restart: finished runs are skipped.
# The queue is experiments/queue.txt, re-read before every run, so lines can be added or reordered on the fly:
#   <model id> <A|B> <same|next> [extra ft_train.py args]   -> training run, done when its metrics.json exists
#   run <done file> <command ...>                           -> any command, done when <done file> exists
# Start: nohup bash experiments/run_all.sh > experiments/logs/queue.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
EXP=experiments/2026-11-23_news-direction
LOGS=experiments/logs
QUEUE=${1:-experiments/queue.txt}
COMMON="--epochs 2 --eval-every 250 --patience 4"
mkdir -p "$LOGS"
if [[ ! -f "$QUEUE" ]]; then
  echo "queue file not found: $QUEUE" >&2
  exit 1
fi
declare -A tried

while true; do
  job=""
  while read -r first second third rest; do
    [[ -z "${first:-}" || "$first" == \#* ]] && continue
    if [[ "$first" == run ]]; then
      key="$second"; done_file="$second"
    else
      key="$(basename "$first" | tr 'A-Z' 'a-z')/${second}_${third}"
      [[ "${rest:-}" =~ --tag[[:space:]]+([^[:space:]]+) ]] && key="${key}_${BASH_REMATCH[1]}"
      done_file="$EXP/$key/metrics.json"
    fi
    [[ -f "$done_file" || -n "${tried[$key]:-}" ]] && continue
    job="$first|$second|$third|${rest:-}"; break
  done < "$QUEUE"
  [[ -z "$job" ]] && break
  IFS='|' read -r first second third rest <<< "$job"
  tried[$key]=1
  log="$LOGS/$(echo "$key" | tr '/' '_' | tr -c 'A-Za-z0-9_.\n-' '_').log"
  echo "$(date '+%F %T') START $key -> $log"
  if [[ "$first" == run ]]; then
    bash -c "$third $rest" > "$log" 2>&1
  else
    # shellcheck disable=SC2086
    bash experiments/spark_run.sh ft_train.py --model "$first" --variant "$second" --target "$third" \
      $COMMON $rest > "$log" 2>&1
  fi
  rc=$?
  echo "$(date '+%F %T') END $key (exit $rc)"
done
echo "$(date '+%F %T') queue empty"
