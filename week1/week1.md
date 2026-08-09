# Week 1 — Blackwell serving baseline + goodput

**Deliverable:** an X thread.
**Done when:** the thread is posted. Not when the hours are used up.

Starting point: vLLM has only ever been used as a black box. So this week deliberately spends its
first evenings building intuition rather than tooling, and the sweep harness is explicitly
*deferred to week 2* so nothing fiddly sits in front of the deliverable.

---

## The question

Written before any measurement, and left unedited afterwards. The point of committing to it in
advance is that the predictions below are falsifiable — not that the answer was known.

**Question.** On this card, serving an 8B model: how far apart are peak output throughput and the
request rate actually servable at an interactive SLO? Is the difference large enough to matter, or
is it a rounding error?

**Why it's worth asking.** Published serving numbers are almost always peak throughput. If the rate
you can serve at a latency bar users would accept is materially lower, then the commonly quoted
number describes a regime nobody operates in. Whether that's true here is an empirical question.

**Predictions, to be checked against the result.**

1. Output throughput rises with offered rate, then plateaus once the GPU saturates.
2. Goodput rises, peaks *below* the throughput plateau, then falls — because continuous batching
   buys throughput by batching harder, and batching harder inflates per-request latency.
3. The peak sits somewhere near 16 req/s. (Little more than a guess; the sweep spans 1–32 to
   bracket it.)
4. TPOT at low load lands near the bandwidth floor, ~9–12 ms. (From
   [`notes/napkin-model.md`](notes/napkin-model.md) — the sharpest prediction here, and the one
   that most cleanly tests whether the cost model is sound.)

**What would falsify the premise.** If goodput tracks throughput and plateaus with it, there is no
gap, prediction 2 is wrong, and the interesting question becomes *why* this card doesn't degrade the
way the model says it should. If goodput never falls by 32 req/s, extend the sweep until the offered
rate exceeds what the server can complete — but report the extension rather than quietly widening
the axis.

**What gets posted depends on which happens.** A large gap is a post about the number most
benchmarks omit. No gap is a post about a model that predicted a collapse that didn't occur — also
publishable, and more interesting than it sounds. **A sweep that produces neither is not a post**;
per `CLAUDE.md`'s honesty valve, that goes in `LOG.md` and the thread is deferred.

**Scope, whatever the result:** one model, one input/output shape, one card, no tensor parallelism,
**untuned vLLM defaults**. A measurement of specific hardware under stated flags — *not* a
Blackwell-vs-H100 claim, and not a tuned-configuration claim. Stating this costs nothing and
forecloses the two most obvious objections.

**Chart to draw either way.** X-axis = offered request rate. One measure per panel (no dual axes):
throughput and SLO-meeting throughput in tok/s on the first, latency against the SLO thresholds on
the second. The shape of the goodput curve is the result, not a foregone conclusion.

*(Outcome and the thread as posted: see [`thread.md`](thread.md) and Results below.)*

---

## Setup

Dedicated venv, **plain vLLM wheel — no source build this week.** The editable
`VLLM_USE_PRECOMPILED=1` build is for the week engine code first needs changing; an evening spent
fighting it now teaches nothing about serving.

Installed and verified: **vLLM 0.26.0**, torch 2.11.0+cu130, Python 3.14.

### Model: `Qwen/Qwen3-8B`

Verified from `config.json` (2026-08-05) — a plain dense GQA decoder, which is what a *comparable*
baseline requires:

```
architectures = ['Qwen3ForCausalLM']
num_hidden_layers = 36, num_attention_heads = 32, num_key_value_heads = 8
head_dim = 128, dtype = bfloat16, max_position_embeddings = 40960
```

KV footprint: `2 × 36 layers × 8 kv_heads × 128 head_dim × 2 bytes` = **144 KiB/token**. Against a
~68 GB arena that's ~495k tokens ≈ ~380 concurrent 1280-token sequences — comfortably more than
`max_num_seqs` allows, so concurrency is scheduler-limited rather than capacity-limited. That is the
regime this week wants.

**Rejected alternatives, and why** (recorded so the choice is defensible in post 2):

- `Qwen/Qwen3.6-27B` — 54 GB of bf16 weights. Decode reads the full weight tensor every step
  *independently of batch size*, so at ~1.8 TB/s the TPOT floor is **~30 ms on an idle card**,
  leaving 1.7× headroom under a 50 ms SLO. Goodput would collapse at ~1 req/s and the chart would
  have no dynamic range. Qwen3-8B's floor is ~9 ms (5× headroom). Also a 51.75 GiB checkpoint
  exceeds available page cache (60 GB box), so every restart pays a full disk read.
- `Qwen/Qwen3.5-9B` — **not a dense LM.** `Qwen3_5ForConditionalGeneration` with a `vision_config`,
  and `layer_types` mixes `full_attention` with `linear_attention` at `full_attention_interval = 4`
  — only 8 of 32 layers hold growing KV; the rest hold fixed-size recurrent state
  (`mamba_ssm_dtype = float32`). vLLM routes it through the hybrid allocator. The collapse would
  still appear (it's driven by batching, not architecture), but the result would be incomparable to
  anything and half the thread would be architecture caveats. Wrong model for a *baseline*.
  - Keep for later: `mtp_num_hidden_layers = 1` means it ships an MTP draft head — a speculative
    decoding target already sitting in the local cache. Noted in `candidates.md`.

### Server, held identical across all runs

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0

vllm serve Qwen/Qwen3-8B \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --max-num-seqs 128 \
  --max-num-batched-tokens 2048
```

| flag | why |
|---|---|
| `VLLM_USE_FLASHINFER_SAMPLER=0` | no CUDA toolkit on this box; FlashInfer JITs at first call and dies. See `candidates.md` |
| `--max-model-len 4096` | ample for 1024-in/256-out; makes the startup concurrency line report the number this workload actually runs at |
| `--gpu-memory-utilization 0.90` | the default, stated explicitly because post 2 promises every flag |
| `--no-enable-prefix-caching` | the confound. On by default in V1. This is post 5 |
| `--max-num-seqs 128` | vLLM's default (`config/scheduler.py:44`), **pinned** — a first-order determinant of where goodput peaks, and defaults drift between versions |
| `--max-num-batched-tokens 2048` | vLLM's default (`config/scheduler.py:42`), **pinned**. Deliberately untuned — see below |

**`--max-num-batched-tokens 2048` is left at the default on purpose.** It fits only ~2 of these
1024-token prompts per prefill step, so prefill chunks across steps under load and TTFT inflates
faster than a tuned server would. That is part of the story in post 4, and "untuned defaults" is the
honest, reproducible baseline. Say so in the thread. It also hands week 3 a subject: *the one flag
that moves TTFT most*.

**What `--max-model-len` does and does not do** (checked in source, because the first guess was
wrong): vLLM V1 allocates KV blocks **on demand**, so a 1280-token request occupies 1280 tokens'
worth of blocks regardless of `max_model_len`. Steady-state KV usage is identical at 4096 and
262144. What `max_model_len` actually controls is (a) a startup pass/fail gate requiring the arena to
hold *one* request at full length (`v1/core/kv_cache_utils.py:750`), (b) a preallocated
`[max_num_seqs, max_model_len/block_size]` int32 block table — ~8 MB at 262144, negligible
(`v1/worker/block_table.py:79-82`), and (c) the reported concurrency figure. So 4096 is a
measurement-hygiene choice, not a memory one.

### Record from the startup log, before running anything

- `GPU KV cache size: N tokens` — sanity-check against the ~495k estimate. Wildly off means the
  memory budget isn't what's assumed, and now is when to find out.
- `Maximum concurrency for 4,096 tokens per request: X` — the quotable number. It is computed *at*
  `max_model_len` tokens per request (`v1/core/kv_cache_utils.py:2177-2181`), which is exactly why
  `max_model_len` is 4096 and not 262144.
- `Chunked prefill is enabled with max_num_batched_tokens=N` — confirms the pin took; `vllm serve`
  can override defaults.

---

## The runs

Seven runs, a few minutes each. **Run by hand** — no harness this week.

**Warmup first, discarded.** Pays CUDA-graph capture and compilation:

```bash
uv run vllm bench serve --backend openai --model Qwen/Qwen3-8B \
  --dataset-name random --random-input-len 1024 --random-output-len 256 \
  --ignore-eos --num-prompts 32 --request-rate inf --seed 999
```

Then the seven measured runs, **same server process throughout — do not restart.** APC is off, so
runs stay comparable without paying seven startup costs. Record that decision in the thread; a warm
KV arena and a cold one are different machines.

```bash
uv run vllm bench serve \
  --backend openai --model Qwen/Qwen3-8B \
  --dataset-name random \
  --random-input-len 1024 --random-output-len 256 --random-prefix-len 0 \
  --ignore-eos \
  --goodput ttft:200 tpot:50 \
  --save-result --save-detailed --result-dir week1/results \
  --num-prompts <N> --request-rate <R> --seed <S> \
  --metadata model=qwen3-8b vllm=0.26.0 apc=off max_num_seqs=128 mnbt=2048 rate=<R>
```

| # | `--request-rate` | `--num-prompts` | `--seed` | purpose |
|---|---|---|---|---|
| 1 | 1 | 200 | 101 | near-idle: best-case TTFT/ITL floor |
| 2 | 2 | 200 | 102 | |
| 3 | 4 | 300 | 103 | |
| 4 | 8 | 500 | 104 | expect goodput still climbing |
| 5 | 16 | 1000 | 105 | expect the peak somewhere near here |
| 6 | 32 | 1500 | 106 | expect collapse |
| 7 | `inf` | 1000 | 107 | saturation — all requests at t=0, peak throughput |

`--num-prompts` scales with rate so every run has ≥60 s of steady state rather than being dominated
by ramp-up. Seeds vary so no two runs share a prompt set.

**Use arrival rate, not fixed concurrency.** `--request-rate` is an open-loop Poisson process
(`--burstiness` defaults to 1.0, true Poisson; it only applies when the rate is finite). Fixed
concurrency is a *closed* loop — slow responses throttle your own offered load, the queue can never
build, and goodput cannot collapse. `--request-rate inf` is the one exception: it fires everything
at t=0.

**Three flags that matter more than they look:**

- **`--save-detailed` is mandatory.** Without it vLLM strips per-request `ttfts`/`itls` from the
  saved JSON (`benchmarks/serve.py:1419`, `ignored_metrics`). With it, the second SLO tier is a
  pandas one-liner; without it, it's a re-run of all seven.
- `--metadata KEY=VALUE` is written into the result JSON. Free provenance.
- `--goodput` is purely post-hoc — thresholds applied to recorded per-request latencies, in
  milliseconds, keys `ttft`/`tpot`/`e2el`.

Flags verified against vLLM **0.26.0** (2026-08-05). Note `--help` alone only prints group names;
`--help=all` prints the actual flags.

Around each run:

```bash
nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv
```

---

## The SLO

Justified, not arbitrary — this is the part that makes it defensible:

- **TTFT ≤ 200 ms** — a normal interactive-chat responsiveness bar.
- **TPOT ≤ 50 ms** — ≈20 tok/s, roughly comfortable reading speed.

Sanity check this against the measured batch-1 TPOT from step 1. Qwen3-8B's bandwidth floor is
~9 ms, so 50 ms leaves ~5× headroom — enough for the curve to climb before it falls. (A model whose
idle floor is already near the SLO produces a curve that only collapses, which is a different post.)

Then re-derive goodput against a **looser tier** (e.g. `ttft:1000 tpot:100`, a "batch-ish" SLO)
from the *same runs*. The point: **goodput depends entirely on the SLO you choose.** Showing two
tiers on one chart is the thing most posts skip, and it's a stronger contribution than the baseline
itself. With `--save-detailed` this is recomputed offline, not re-measured.

---

## Confounds, and what's done about each

The generic list lives in `CLAUDE.md`. What actually bites *this* setup:

- **Prefix caching across runs.** The real trap, and subtle. The `random` dataset generates distinct
  prompts *within* a run, so nothing looks wrong. But a fixed `--seed` produces the *same prompt set*
  on every run — so runs 2–7 hit a warm prefix cache and report a spuriously fast TTFT while run 1
  doesn't. A monotonic bias across the sweep that looks exactly like a real effect. Three independent
  fixes exist; this week uses the first two:
  1. `--no-enable-prefix-caching` on the server.
  2. Vary `--seed` per run.
  3. `POST /reset_prefix_cache` between runs — an endpoint on any vLLM server. This is what
     `vllm bench sweep` does by default (`benchmarks/sweep/server.py:15`, alongside
     `/reset_mm_cache` and `/reset_encoder_cache`). Worth knowing: the vLLM authors hit the same
     confound and chose reset-between-runs over disabling APC.
  - Note what disabling APC does *not* do: it does not make generation re-prefill the prompt per
    token. Intra-request paged KV is always on — a request prefills once, then each new token
    appends its K/V and attends to what's stored. APC is purely *cross-request* prefix reuse. At
    1024 random tokens the only shared prefix is the chat template (~20–30 tokens, 1–2 blocks), so
    turning it off costs ≈nothing here while removing the contamination entirely.
  - Measuring what APC actually *buys* is a separate experiment and a good candidate week — the
    `prefix_repetition` dataset is purpose-built for it. Don't conflate it with the baseline.
- **`--reasoning-parser` silently corrupts the measurement.** Do not enable it on the benchmark
  server. `vllm bench serve`'s chat backend reads only `choices[0]["delta"]["content"]`
  (`benchmarks/lib/endpoint_request_func.py:404`) and never looks at `reasoning_content`. With a
  reasoning parser active, thinking tokens arrive as `content: None`, so measured TTFT becomes "time
  to first *post-thinking* token" — inflated by the entire reasoning trace — and those tokens don't
  count toward throughput. Plausible-looking numbers that are wrong. Caught before any measurement;
  worth a line in post 5 alongside the APC trap.
- **Warmup.** First requests pay CUDA-graph capture and compilation. The discarded warmup run above.
- **Clock drift.** Seven runs back to back heats the card. Read clocks before and after each run and
  check for monotonic decline across the sweep. Clock locking needs privileges this box doesn't
  have, so detection is the best available option — and a drift check in the thread is itself a
  credibility signal.
- **Server restarts.** Decided: **don't restart.** Recorded above.
- **Untuned defaults.** Not a confound so much as a scope limit, but state it: `max_num_seqs=128`
  and `max_num_batched_tokens=2048` are vLLM defaults, and both bound the curve. Someone will point
  this out; answering it before they do is free.

---

## Napkin model — predict before you measure

Full derivation in [`notes/napkin-model.md`](notes/napkin-model.md). Standing practice, not a
week-1 task: **write a predicted TTFT / TPOT / throughput before each run executes.** A prediction
from a model is wrong in a *specific way*, so the residual names the term you misunderstood; a run
you merely read teaches nothing.

The model in one line each:

- **Decode is bandwidth-bound.** `t_decode = W/BW + (B·L·k)/BW + c`. The weight term is a floor
  independent of `B` — which is why batching is nearly free for throughput.
- **Prefill is compute-bound.** `t_prefill(n) = FLOPs(n)/C`, with `FLOPs(n) = 2·P·n` plus an
  attention term quadratic in `n`.
- **The scheduler couples them.** Decodes claim the token budget first, prefill gets `T − B`
  (`v1/core/sched/scheduler.py:445,471,669`). So rising load shrinks prefill capacity *and* grows the
  queue — **TTFT degrades superlinearly while throughput only asymptotes.** That divergence is this
  week's post, derived rather than observed.

Two constants must be calibrated from measurement (`C`, `c`); the rest come from config and the spec
sheet. Both are recoverable from `upload_result/warmup_bench.txt` before run 1.

---

## Steps

Loose ordering, not a schedule. Stop when the post ships.

0. **Unblock, first thing** — both have external latency:
   - `uv run hf download Qwen/Qwen3-8B` (~16 GB; only `config.json` is local as of 2026-08-05).
   - `git init` this folder. The thread links a repo; decide what goes public *before* the thread is
     otherwise ready. It's the item most likely to cause hesitation at the moment of posting.
1. **Make the metrics concrete.** Serve the 8B. Send *one* streaming request and watch it:
   time-to-first-chunk **is** TTFT, inter-chunk gaps **are** ITL. **Write down batch-1 TPOT** — it's
   the sanity check on the SLO. Then 8 concurrent, and watch TTFT inflate while aggregate throughput
   climbs. Keep `/metrics` open alongside — `vllm:num_requests_running`,
   `vllm:num_requests_waiting`, `vllm:gpu_cache_usage_perc`. Watching those move is the cheapest
   introduction to continuous batching that exists.
2. **One benchmark run by hand.** Learn `vllm bench serve`'s flags and output, and note what it does
   *not* measure. Confirm the SLO.
3. **The seven runs.** Plus clock readings around each.
4. **Plot, then post.** Read the seven JSONs, plot two lines (three with the loose SLO tier). ~20
   lines of matplotlib — the only code this week needs. Optionally try `vllm bench sweep plot` on
   `results/` first (`--dry-run` to check it before investing); `request_goodput` is in the
   result JSON, so `--curve-by` may just work.

**If the curve doesn't collapse:** `max_num_seqs=128` caps the running batch well below KV capacity,
so the collapse may be softer than expected. Push the rate to 64 and 128 before touching any other
flag.

---

## Results

### Server startup (2026-08-08 15:41, raw log in `upload_result/server_start.txt`)

All six flags took — `non-default args` confirms `max_model_len=4096`,
`gpu_memory_utilization=0.9`, `enable_prefix_caching=False`, `max_num_batched_tokens=2048`,
`max_num_seqs=128`, `host=0.0.0.0`.

```
GPU KV cache size: 508,192 tokens
Maximum concurrency for 4,096 tokens per request: 124.07x
Graph capturing finished in 2 secs, took 0.29 GiB
```

Both are quotable in post 2. Worth noting for the "scheduler-limited, not capacity-limited" claim:
at the full 4096-token context the KV arena allows 124 concurrent — slightly *below*
`max_num_seqs=128`. At this week's ~1280-token working length it allows far more, so
`max_num_seqs` is the binding constraint as intended. Check this arithmetic by hand.

### Warmup (raw in `upload_result/warmup_bench.txt`) — discarded as data, used for calibration

32 requests at `rate=inf`. Exact token counts (`32768` in = 32×1024, `8192` out = 32×256) confirm
both `--backend openai` (no chat-template overhead) and `--ignore-eos`. `Failed requests: 0`.

Use this file to calibrate `C` and `c` per [`notes/napkin-model.md`](notes/napkin-model.md) §4
before run 1.

<!-- fill in as runs complete: per-rate numbers, the gap, clock drift -->

## Notes / what surprised me

<!-- raw material for post 5 and for planning week 2 -->
<!-- DAILY NOTES — raw, unedited. Clean up once the post ships. -->

### 2026-08-01 (day 1) — environment only, no serving numbers

Zero progress on step 1. The whole evening went to environment setup, and one real finding fell out
of it that's worth a week-2 post.

**Done**

- `uv init` + `uv add vllm` → `vllm 0.26.0`, `torch 2.11.0+cu130`. First failure was self-inflicted:
  `uv init` set `requires-python = ">=3.14"` (newest installed Python) while `.python-version`
  said `3.12`, so the two disagreed out of the box.
- Verified Blackwell support: `torch.cuda.get_device_capability() == (12, 0)`, `sm_120` present in
  `torch.cuda.get_arch_list()`. Plain PyPI wheel is correct for this card — no special index.
- **Migrated the project 3.12 → 3.14.** `uv lock` resolved all 202 packages, including every
  compiled dep (`triton`, `tilelang`, `flashinfer`, `xgrammar`). Smoke test on `opt-125m` passes.
- Hit `RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`
  during warmup. Worked around with `VLLM_USE_FLASHINFER_SAMPLER=0`.

**Learned**

- **Driver ≠ runtime ≠ toolkit.** Have the driver (595.84) and the CUDA runtime libs bundled in the
  torch wheel; do *not* have the toolkit (`nvcc` + headers). Almost everything ships precompiled
  kernels, so nothing notices — until something JIT-compiles at runtime.
- **Why only FlashInfer crashed, and DeepGEMM didn't.** Two different designs. DeepGEMM is gated
  behind `has_deep_gemm()`, a presence probe that fails at *import* (bare
  `assert cuda_home is not None`, `vllm/third_party/deep_gemm/__init__.py:117`) and falls back to
  Cutlass silently. FlashInfer's sampler checks env var + compute capability, *assumes* the package
  works, and only discovers it needs `nvcc` at the first kernel call inside warmup — unguarded, so
  fatal. Probe-before-use vs. assume-then-JIT.
- **`abi3` wheels vs version-specific ones** is why 3.14 was even viable. vLLM ships
  `cp38-abi3` (one build works on any CPython ≥3.8); torch ships per-version `cp314` wheels. I
  initially assumed vLLM had no 3.14 wheels — wrong, and the wheel filename says so directly.
- uv shares interpreters and the wheel cache globally (`~/.cache/uv`, hardlinked into each venv);
  the `.venv` and `uv.lock` are the only per-project isolation. Same story as the HF cache
  (141 GB, shared, keyed by repo id) — the 3.14 migration re-downloaded nothing model-side.
- **Diagnostic heuristic worth keeping:** fluent-but-wrong generation means the stack is fine
  (`opt-125m` said "getting its final proposal from a country"). Broken numerics produce repeated
  tokens / `!!!!` / empty strings, not grammar.

**The finding — candidate week-2 post**

The missing toolkit disables *two* paths, not one. The sampler is irrelevant to throughput. But
DeepGEMM's FP8 GEMM falling back to `CutlassFp8BlockScaledMMKernel` means **every FP8 number I've
measured on this card ran on a fallback kernel** — the FoT project has been quietly doing this since
July. So my own published Blackwell numbers are probably understated. Captured in
[`candidates.md`](../candidates.md) as "I understated my own hardware", with the A/B design and the
must-not-compare-against-stale-baselines confound.

**Blocking tomorrow — both have wall-clock latency, start them first**

1. **`Qwen/Qwen3-8B` is not in the HF cache.** Cache has Qwen3.5-9B, Qwen3-14B-FP8, Qwen3-32B,
   Qwen3-32B-FP8, DeepSeek-R1-7B, opt-125m — no 8B. Kick off the download before anything else, or
   switch the week to `Qwen/Qwen3.5-9B` which is already local. Deciding this costs nothing now and
   costs an evening if discovered at step 3.
2. **This file's flags were verified against vLLM 0.25.1; installed is 0.26.0.** Re-check
   `vllm bench serve --help=all` before trusting the run table.

Also still open from the setup: `VLLM_USE_FLASHINFER_SAMPLER=0` needs to live in a serve script, not
be retyped — and `git init` for this folder is still not done (week 1's post links a repo).

### 2026-08-05 (day 2) — no runs yet; config settled and three traps caught by reading source

Still zero serving numbers. The evening went to deciding the configuration, and reading vLLM's own
source caught three things that would have cost real time or produced wrong numbers. That reading
habit is itself the week's transferable lesson — for a lane-2 (framework internals) target, reading
the framework before writing around it *is* the skill.

**Model decision.** Resolved day 1's blocker by *not* taking the shortcut. `Qwen3.5-9B` is local but
is a hybrid-linear-attention VLM (see Setup) — wrong for a baseline. `Qwen3.6-27B` is local but its
bf16 weights put the TPOT floor at ~30 ms, leaving no room under a 50 ms SLO. Went back to
`Qwen3-8B` and started the download. **The lesson: "already in the cache" is not a model-selection
criterion.** Two evenings of avoiding a 16 GB download nearly bought an uninterpretable chart.

**Three traps caught by reading source, not by running anything**

1. **`--reasoning-parser` would have corrupted TTFT.** `serve.sh` had it (copied from an unrelated
   script). The bench client only reads `delta.content`, never `reasoning_content`
   (`benchmarks/lib/endpoint_request_func.py:404`), so every thinking token would have been
   invisible and TTFT would have measured time-to-first-*post*-thinking-token. This is the *same
   class of error* as the APC trap — a plausible number produced by a mechanism I hadn't looked at.
   Post 5 now has two examples instead of one, which makes the point about mechanism rather than
   about one flag.
2. **`--save-detailed` is not optional.** Per-request `ttfts`/`itls` are stripped from saved results
   without it (`benchmarks/serve.py:1419`). Would have discovered this at step 4, with the
   two-tier-SLO chart requiring a full re-run of the sweep.
3. **`vllm bench sweep` already exists** — `{serve, serve_workload, startup, plot, plot_pareto}`.
   `sweep serve` starts one server per serve-config and iterates bench-configs against it
   (`benchmarks/sweep/serve.py:267`), repeats each combo `--num-runs 3`, supports `--resume` and
   `--dry-run`, POSTs the cache-reset endpoints between runs, and writes `summary.csv`. `sweep
   plot_pareto` plots tokens/s/user against tokens/s/GPU — the latency/throughput frontier this
   week's claim is about. **Week 2's planned "sweep harness" was going to be a reimplementation.**
   Rewritten in `week2.md`.
   - Still running week 1 by hand: seven commands is ~40 min, learning `sweep`'s param-JSON and
     `link_vars` semantics is an evening, and it would hide exactly what step 2 exists to learn.

**Corrected a wrong belief about `--max-model-len`.** Assumed a large value reserved KV per
sequence. It doesn't — V1 allocates blocks on demand and steady-state usage is identical at 4096 and
262144. The real effects are a startup feasibility gate, ~8 MB of block table, and which concurrency
figure gets logged. Keeping 4096 for log clarity, not for memory. Worth remembering as a general
caution: "flag sounds expensive" is not evidence, and the arithmetic that *seems* to confirm it
(8.6 GB for a full-length sequence) was a real number answering a different question.

**Also learned**

- `max_num_seqs` defaults to **128** (`config/scheduler.py:44`) and `max_num_batched_tokens` to
  **2048** (`config/scheduler.py:42`). Both bound the goodput curve and neither was in the original
  flag list. Now pinned explicitly rather than inherited.
- Request logging is **opt-in** in 0.26.0 (`--enable-log-requests`, default off), so per-request log
  spam at 32 req/s isn't a confound to defend against.
- The "Filesystem type for checkpoints / Available RAM" startup line is about **system RAM, not
  VRAM**, and on a local FS (EXT4) it's informational only — auto-prefetch is off regardless of
  whether the checkpoint fits (`model_executor/model_loader/weight_utils.py:870`). It gates nothing.
  But it does surface a real consequence of the 60 GB/96 GB imbalance already noted in `CLAUDE.md`:
  checkpoints above ~45 GiB can't stay page-cache resident, so large-model iteration pays a full
  disk read on every restart. Independent argument for the RAM upgrade.
