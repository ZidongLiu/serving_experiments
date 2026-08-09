#!/usr/bin/env bash
#
# Week 1 — the seven measured runs.
#
# Runs against an ALREADY-RUNNING server (../../serve.sh). One server process for all seven
# runs: do NOT restart between them. Prefix caching is off on the server side, so a warm KV
# arena stays comparable across runs and we don't pay seven startup costs.
#
# Send the discarded warmup run before this script (CUDA-graph capture + compilation).
#
# Usage:  bash week1/src/benchmark.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEEK_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$WEEK_DIR")"
RESULT_DIR="$WEEK_DIR/results"
CLOCKS_CSV="$RESULT_DIR/clocks.csv"

MODEL="Qwen/Qwen3-8B"
VLLM_VERSION="0.26.0"
ISL=1024
OSL=256
HOST=localhost
PORT=8000

# Pinned server settings, echoed into --metadata so every result JSON is self-describing.
MAX_NUM_SEQS=128
MAX_NUM_BATCHED_TOKENS=2048

cd "$REPO_ROOT"
mkdir -p "$RESULT_DIR"

# The full experiment, declared in one place and ordered by rate.
#
#   num_prompts scales with rate so every run has >=60s of steady state rather than being
#   dominated by ramp-up. Seeds are unique per run so no two runs share a prompt set
#   (belt-and-braces alongside --no-enable-prefix-caching on the server). Seeds are
#   non-monotonic because 108-111 were added after 101-107 had already run — they are labels,
#   not a sequence, and the originals must keep their seeds so existing results stay valid.
#
# Measured service capacity is ~9.3 req/s, so rho = rate/9.3 crosses 1 between rates 9 and 10.
# Rates 9-14 exist to resolve that knee. Above rho=1 there is no steady state: mean TTFT grows
# with run length rather than converging, so those TTFT values describe (server, num_prompts),
# not the server. Goodput ~= 0 is the robust claim there.
#
# rate  num_prompts  seed     rho     note
RUNS=(
  "1    200  101"   #  0.11   near-idle: best-case TTFT/ITL floor
  "2    200  102"   #  0.22
  "4    300  103"   #  0.43
  "8    500  104"   #  0.86   goodput peak (5.75 req/s)
  "9    1000 108"   #  0.97   just below the wall — slowest to converge, so run it longest
  "10   500  109"   #  1.08   just above the wall
  "12   500  110"   #  1.29
  "14   500  111"   #  1.51
  "16   1000 105"   #  1.72   collapse confirmed
  "32   1500 106"   #  3.44
  "inf  1000 107"   #   inf   saturation: all requests at t=0, peak throughput
)

# Skip a run if a result JSON already exists for that rate, so re-invoking only fills gaps.
# Set FORCE=1 to re-run everything.
FORCE="${FORCE:-0}"

result_exists() {  # $1 = rate
  local pattern
  if [[ "$1" == "inf" ]]; then
    pattern="$RESULT_DIR/openai-infqps-*.json"
  else
    pattern="$RESULT_DIR/openai-${1}.0qps-*.json"
  fi
  compgen -G "$pattern" > /dev/null
}

record_clocks() {  # $1 = rate, $2 = phase (before|after)
  local sample
  sample=$(nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw \
                      --format=csv,noheader,nounits | tr -d ' ')
  echo "$1,$2,$(date -Is),$sample" >> "$CLOCKS_CSV"
}

# Fail before run 1 rather than after seven broken runs.
if [[ "$(curl -s -o /dev/null -w '%{http_code}' "http://$HOST:$PORT/health" || true)" != "200" ]]; then
  echo "ERROR: no healthy server at $HOST:$PORT — start serve.sh first" >&2
  exit 1
fi

if [[ ! -f "$CLOCKS_CSV" ]]; then
  echo "rate,phase,timestamp,clocks_sm_mhz,temp_c,power_w" > "$CLOCKS_CSV"
fi

for run in "${RUNS[@]}"; do
  read -r R N S _ <<< "$run"

  if [[ "$FORCE" != "1" ]] && result_exists "$R"; then
    echo "skip  rate=$R — result JSON already present (FORCE=1 to re-run)"
    continue
  fi

  echo
  echo "=================================================================="
  echo "  rate=$R  num_prompts=$N  seed=$S"
  echo "=================================================================="

  record_clocks "$R" before

  uv run vllm bench serve \
    --backend openai --model "$MODEL" \
    --host "$HOST" --port "$PORT" \
    --dataset-name random \
    --random-input-len "$ISL" --random-output-len "$OSL" --random-prefix-len 0 \
    --ignore-eos \
    --goodput ttft:200 tpot:50 \
    --save-result --save-detailed --result-dir "$RESULT_DIR" \
    --num-prompts "$N" --request-rate "$R" --seed "$S" \
    --metadata model=qwen3-8b vllm="$VLLM_VERSION" apc=off \
               max_num_seqs="$MAX_NUM_SEQS" mnbt="$MAX_NUM_BATCHED_TOKENS" \
               rate="$R" num_prompts="$N" seed="$S" \
    2>&1 | tee "$RESULT_DIR/bench_rate${R}.log"

  record_clocks "$R" after
done

echo
echo "done — result JSONs in $RESULT_DIR"
echo "       clocks in $CLOCKS_CSV (check for monotonic decline across runs)"
