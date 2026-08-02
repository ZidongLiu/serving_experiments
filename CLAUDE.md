# master_serving

Workspace for a deliberate deep dive into **LLM inference/serving at scale**, with the goal of
transitioning into a full-time LLM serving role.

## Context

Owner is a software engineer currently doing AI research (3 papers accepted at ICLR/ICML).
Strong on experimental rigor, ramping on systems/inference internals.

**Time budget: ~2 h every night** (~14 h/week). Implication: wall-clock is scarcer than effort,
so kick off anything with external latency (PR review, GPU rentals, long runs) as early as
possible. Long sweeps must run unattended and resumable — never babysat within one sitting.

## Workflow

**Owner writes and runs all code in this repo manually.** Claude's role is design, explanation,
review, and analysis of results — not implementation. Do not scaffold or write code unless
explicitly asked.

## Hardware (verified 2026-07-30)

- **GPU:** 1× NVIDIA RTX PRO 6000 Blackwell, 96 GB VRAM (driver 595.84, CUDA 13.2)
- **CPU/RAM:** 24 cores, 60 GB system RAM
- **Disk:** 1.6 TB free on `/`

Implications:
- Native **FP4 (nvfp4/mxfp4)** support on Blackwell — new, under-covered, high-leverage niche.
- 96 GB allows 70B at FP8/INT4, MoE (e.g. Qwen3-30B-A3B), or large-KV-cache experiments.
- **Gap:** single GPU, no NVLink → no local multi-GPU TP/PP/EP or disaggregated prefill/decode.
  Plan: design locally, rent 8×H100 (Lambda/RunPod/Together) for focused weekends.
- **Suggested upgrade:** 60 GB system RAM is lopsided vs 96 GB VRAM; 128–256 GB unblocks CPU KV
  offload / LMCache-style tiering.
- **Not available: MIG.** Verified 2026-07-30 — `mig -lgip` reports no MIG-supported devices and
  `mig.mode.current` is `N/A` (not `Disabled`). MIG ships only on datacenter parts, never on the
  workstation line. Multi-instance routing/scheduling experiments are off the table locally.

## Model choice

No single fixed model. Different tricks solve different challenges, so the model is chosen per
topic. Rule: **pick the smallest model that still exhibits the phenomenon** — iteration speed is
the scarce resource. Pin the exact checkpoint per experiment and record it in the results;
comparing across models is not a comparison.

| Topic | What the model must provide | Class |
|---|---|---|
| Scheduler, continuous batching, chunked prefill, CUDA graphs | fast startup, KV headroom for high concurrency | 8B dense |
| Prefix caching / APC | nothing special — the *workload* carries the shared prefixes | 8B dense |
| Bandwidth-bound decode, FP8 vs NVFP4 weights | weights big enough to dominate decode; a quantized checkpoint or a `llm-compressor` path | 32B FP8 / NVFP4 |
| KV-cache quant, large-KV behavior | large KV footprint — long context, and GQA-vs-MHA matters | long-context, 32k+ |
| Speculative decoding (EAGLE / n-gram) | **hard constraint:** target needs published draft or EAGLE weights — let availability pick the target, not the reverse | whatever has heads |
| MoE routing / expert behavior | an MoE checkpoint fitting in 96 GB | Qwen3-30B-A3B |

## Chosen lane

Three hiring lanes exist: (1) kernel/perf engineering, (2) serving-framework internals,
(3) inference platform/SRE. **Target lane 2 (framework internals), with a foot in lane 1.**
Best fit for a research background, and mostly reachable on a single-node lab.

**Primary credential to build: merged PRs in vLLM and/or SGLang.** In this field those outrank
papers for hiring. Ramp: `good first issue` → correctness bugs with clean repros → features.

## Cadence

**Plan one week at a time, at the start of that week.** No long-range schedule — each week is
planned from what actually got done and learned, using `LOG.md` and `experiments/` as the record.

**Every week ships a deliverable: an X post.** The deliverable is the forcing function; without a
public claim the work drifts. Honesty valve: if a week turns out to be pure infrastructure with no
defensible finding, say so in `LOG.md` and defer the post — do *not* pad a post with setup work or
a number you can't defend.

**Definition of done: the post ships.** Hours are not the unit of work. Finishing in three evenings
means the week is done — rest, don't backfill scope. Working past the deliverable because time
remains is how this project dies. Corollary for planning: each week is the *minimal path to the
post*. Anything not required for the post is explicitly a separate week's work, never a
prerequisite bolted on in front of the deliverable.

## Direction (not a schedule)

Rough ordering only, with no dates attached. Deliberately vague past the current week.
Measure → get inside the vLLM V1 engine (scheduler, paged KV, prefix caching, CUDA graphs) →
first merged PR → distributed work on rented hardware if and when it's earned.

## Tracking

Everything lives in this folder so week N can be planned from evidence.

- `experiments/NNN-slug/` — one directory per experiment. A `README.md` stating **the question**,
  the exact setup (model, vLLM version, server flags), the result, and *what was surprising*.
  Raw results as data files next to it.
- `LOG.md` — one entry per week: what was done, what was learned, what the post claimed.
  This is the input to planning the next week.

## Weekly plans

Each week gets its own file — `week1.md`, `week2.md`, … — holding that week's post spec, runs,
confounds, steps, and a results/notes section filled in as the week goes. Keep this file for
standing context; keep the week-specific detail out of it.

**Current: [`week1.md`](week1.md)** — Blackwell serving baseline + goodput. Post claim: peak
throughput and the rate actually servable at an interactive SLO are very different numbers, and the
gap is the finding. Sweep harness explicitly deferred to week 2.

**Candidate: [`week2.md`](week2.md)** — not yet planned. Holds the CUDA-toolkit item: this box has
no toolkit, so FlashInfer's sampler and DeepGEMM's FP8 GEMM are both bypassed, and every FP8 number
measured so far ran on a Cutlass fallback. Candidate post: "I understated my own hardware."

## Positioning

Lead with systems artifacts, not the papers. Use the papers as evidence of *rigor* — experiment
design, confound control, not fooling yourself with a benchmark.

## Known benchmark confounds

Check these before trusting any number. The first one is the most common way serving benchmarks
lie:

- **Prefix caching is on by default in vLLM V1.** Reusing the same prompt across runs produces
  cache hits and a spuriously fast TTFT. Either disable it or randomize prompt prefixes — and say
  which you did.
- **Warmup.** The first requests pay CUDA-graph capture and compilation. Discard them explicitly.
- **Output length.** Use `ignore_eos` with a fixed output length, or ITL averages over runs of
  different lengths and means nothing.
- **Clock drift.** A long sweep heats the GPU and clocks fall. Record clocks per run and check for
  monotonic drift across the sweep. (Locking clocks needs privileges this box doesn't have.)
- **Server state.** Decide and record whether the server restarts between configs; a warm KV arena
  and a cold one are different machines.
