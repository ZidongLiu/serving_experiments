# Week 1 — Blackwell serving baseline + goodput

**Deliverable:** an X thread.
**Done when:** the thread is posted. Not when the hours are used up.

Starting point: vLLM has only ever been used as a black box. So this week deliberately spends its
first evenings building intuition rather than tooling, and the sweep harness is explicitly
*deferred to week 2* so nothing fiddly sits in front of the deliverable.

---

## The post

**Claim.** On this card, serving an 8B model, peak throughput and the request rate actually
servable at an interactive SLO are very different numbers. The gap is the finding.

**Headline number.** "Peak throughput says X tok/s. At a 200 ms TTFT / 50 ms TPOT SLO, this card
serves Y req/s." The saturation point sits well below where the throughput curve flattens.

**The chart.** One chart. X-axis = request rate (req/s). Two lines:

- total output throughput (tok/s) — climbs, then plateaus
- **goodput** (req/s meeting the SLO) — climbs, peaks, then *collapses*

The collapse is the story. A curve that only climbs has no post in it — if goodput hasn't fallen
over by 32 req/s, push the rate higher until it does.

**Thread outline (~6 posts).**

1. Hook: the gap number.
2. Setup: exact card, model, vLLM version, every server flag. This is what buys credibility.
3. The chart.
4. Why the curves diverge: continuous batching buys throughput by batching harder, and batching
   harder inflates per-request latency. Throughput and latency aren't a tradeoff you tune between
   — they're one knob read two ways.
5. **The confound confession** — prefix caching is on by default and would have handed over a fake
   TTFT. What went wrong, how it was caught, what was done about it.
6. Repo link, what's next.

Post 5 is the differentiator. "Here's the trap I nearly fell into" travels further than a curve,
and it's the honest-researcher positioning rather than a benchmark flex.

**Scope the claim explicitly:** one model, one input/output shape, one card, no tensor parallelism.
This is a measurement of specific hardware under stated flags — *not* a Blackwell-vs-H100 claim.
Say so in the thread; it costs nothing and forecloses the most obvious objection.

---

## Setup

Dedicated venv, **plain vLLM wheel — no source build this week.** The editable
`VLLM_USE_PRECOMPILED=1` build is for the week engine code first needs changing; an evening spent
fighting it now teaches nothing about serving.

Model: `Qwen/Qwen3-8B` — ungated on HF (no license click-through to stall you), current, 8B dense,
and small enough in 96 GB to leave a large KV arena for high-concurrency runs. Qwen3 has hybrid
thinking mode, which would perturb output lengths; `--ignore-eos` with a fixed output length makes
that moot here. If you'd rather dodge the wrinkle entirely, `Qwen/Qwen2.5-7B-Instruct` is the most
boring standard alternative.

Server, held **identical across all runs**:

```
vllm serve Qwen/Qwen3-8B \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching
```

`--max-model-len 4096` is ample for 1024-in/256-out and leaves maximum KV headroom, so concurrency
is limited by the scheduler rather than by cache capacity. Record the reported KV cache size at
startup — it's a number the post should quote.

---

## The runs

Seven runs, a few minutes each. **Run by hand** — no harness this week.

| # | `--request-rate` | purpose |
|---|---|---|
| 1 | 1 | near-idle: best-case TTFT/ITL floor |
| 2 | 2 | |
| 3 | 4 | |
| 4 | 8 | expect goodput still climbing |
| 5 | 16 | expect the peak somewhere near here |
| 6 | 32 | expect collapse |
| 7 | `inf` | saturation — all requests at t=0, peak throughput |

```
vllm bench serve \
  --backend openai-chat --model Qwen/Qwen3-8B \
  --dataset-name random \
  --random-input-len 1024 --random-output-len 256 --random-prefix-len 0 \
  --ignore-eos \
  --num-prompts 200 \
  --request-rate 8 \
  --goodput ttft:200 tpot:50 \
  --seed <VARY PER RUN> \
  --save-result --result-dir results/week1 \
  --metadata model=qwen3-8b vllm=<version> apc=off rate=8
```

Flags above verified against vLLM **0.25.1**. A fresh venv may install newer — re-check
`vllm bench serve --help=all` if anything errors. (Note `--help` alone only prints group names;
`--help=all` prints the actual flags.)

Two flags worth knowing about:

- `--metadata KEY=VALUE` is written into the saved result JSON. Free provenance — use it.
- `--burstiness` defaults to 1.0, which is a true Poisson process. Leave it; just know it's there
  and that it only applies when `--request-rate` is finite.

`--num-prompts 200` at rate 1 means a ~200 s run; scale it up at high rates so each run has enough
requests to be meaningful rather than dominated by ramp-up.

---

## The SLO

Justified, not arbitrary — this is the part that makes it defensible:

- **TTFT ≤ 200 ms** — a normal interactive-chat responsiveness bar.
- **TPOT ≤ 50 ms** — ≈20 tok/s, roughly comfortable reading speed.

Then re-derive goodput against a **looser tier** (e.g. `ttft:1000 tpot:100`, a "batch-ish" SLO)
from the *same runs* — `--goodput` is just a threshold applied to recorded per-request latencies,
so this costs one extra invocation, not extra measurement. The point: **goodput depends entirely
on the SLO you choose.** Showing two tiers on one chart is the thing most posts skip, and it's a
stronger contribution than the baseline itself.

---

## Confounds, and what's done about each

The generic list lives in `CLAUDE.md`. What actually bites *this* setup:

- **Prefix caching across runs.** This is the real trap here, and it's subtle. The `random` dataset
  generates distinct prompts *within* a run, so nothing looks wrong. But a fixed `--seed` produces
  the *same prompt set* on every run — so runs 2–7 hit a warm prefix cache and report a
  spuriously fast TTFT, while run 1 doesn't. That's a monotonic bias that looks exactly like a real
  effect. Two independent fixes, and this week uses both: `--no-enable-prefix-caching` on the
  server, and vary `--seed` per run.
  - Measuring what APC actually *buys* is a separate experiment and a good candidate for a later
    week. Don't conflate it with the baseline.
- **Warmup.** First requests pay CUDA-graph capture and compilation. Send a handful of throwaway
  requests after startup before the first measured run.
- **Clock drift.** Seven runs back to back heats the card. Record
  `nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw` before and after each run and check
  for monotonic decline across the sweep. Clock locking needs privileges this box doesn't have, so
  detection is the best available option — and a drift check in the thread is itself a credibility
  signal.
- **Server restarts.** Decide once: restart between runs, or don't. Record which. A warm KV arena
  and a cold one are different machines. (Recommendation: don't restart, but *do* disable APC, so
  runs stay comparable without paying seven startup costs.)

---

## Steps

Loose ordering, not a schedule. Stop when the post ships.

1. **Make the metrics concrete.** Venv + wheel + serve the 8B. Send *one* streaming request and
   watch it: time-to-first-chunk **is** TTFT, inter-chunk gaps **are** ITL. Then 8 concurrent, and
   watch TTFT inflate while aggregate throughput climbs. Keep `/metrics` open alongside —
   `vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:gpu_cache_usage_perc`. Watching
   those move is the cheapest introduction to continuous batching that exists.
2. **One benchmark run by hand.** Learn `vllm bench serve`'s flags and output, and note what it
   does *not* measure. Write down the SLO.
3. **The seven runs.** Plus clock readings around each.
4. **Plot, then post.**

**Also, early — not at the end:** this folder isn't under git, and the thread links a repo.
Decide what goes public and initialize it *before* the thread is otherwise ready. It's the one
item this week with external latency, and it's the thing most likely to cause hesitation at the
moment of posting.

---

## Results

<!-- fill in as runs complete: KV cache size at startup, per-rate numbers, the gap, clock drift -->

## Notes / what surprised me

<!-- raw material for post 5 and for planning week 2 -->
