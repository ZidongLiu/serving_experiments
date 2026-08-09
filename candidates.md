# Parked candidates

Ideas with design work already done, not committed to any week. Recorded so they aren't
rediscovered from scratch. Promote one into `weekN/weekN.md` when a week chooses it.

Originally written as `week2.md` (2026-08-01, revised 08-05); week 2 went to the cost-model
question instead, so this became the holding pen. Nothing here has been run.

---

## Carried in from week 1

- ~~**Sweep harness.** Week 1 explicitly deferred it and ran seven benchmarks by hand.~~
  **Dead — vLLM already ships one.** See below. Week 1 still ran by hand on purpose (learning the
  flags was the point); there is nothing left to *build*.
- **Two example traps for the "confound confession" voice**, now established in week 1's post 5:
  prefix caching on by default, and `--reasoning-parser` making the bench client blind to streamed
  tokens. This post continues that thread rather than repeating the trick.

### The harness does not need building (verified 2026-08-05, vLLM 0.26.0)

`vllm bench sweep` — subcommands `{serve, serve_workload, startup, plot, plot_pareto}`:

- **`sweep serve`** takes `--serve-cmd` and `--bench-cmd` plus `--serve-params` / `--bench-params`
  JSON files and iterates their Cartesian product. Structure
  (`benchmarks/sweep/serve.py:267`) is one server per serve-config, all bench-configs run against
  that same server — exactly the "don't restart between rates" discipline week 1 arrived at by hand:

  ```python
  for serve_comb in serve_params:
      with server_ctx(...) as server:
          for bench_comb in bench_params:
              run_comb(..., num_runs=num_runs)
  ```

- `--num-runs` defaults to **3**, so run-to-run variance comes free — a rigor upgrade over week 1's
  single-shot runs, and worth having for an A/B where the effect may be small.
- `--resume` runs only combinations with no output file yet. Satisfies `CLAUDE.md`'s
  unattended-and-resumable requirement without writing it.
- `--dry-run` prints the commands without executing. Use this first, always.
- Between runs it POSTs `/reset_prefix_cache`, `/reset_mm_cache`, `/reset_encoder_cache`
  (`benchmarks/sweep/server.py:15`), overridable via `--after-bench-cmd`.
- `--link-vars` couples serve and bench parameters (e.g. `max_num_seqs=max_concurrency`) so invalid
  combinations are skipped rather than run.
- Writes `summary.csv` across the whole sweep.
- **`sweep plot`** draws curves with `--curve-by` / `--fig-by` / `--col-by` / `--filter-by` /
  `--bin-by`. **`sweep plot_pareto`** plots tokens/s/user against tokens/s/GPU — the
  latency-vs-throughput frontier.

Implication for this week: the effort budget that was earmarked for harness-building goes into the
experiment instead. **Learning `sweep`'s parameter-JSON semantics is now a real task, but it replaces
a larger one.** Budget one sitting for `--dry-run` iteration before trusting a long unattended run.

## New candidate: the CUDA toolkit item

### Where it came from

Setting up this repo's venv, `vllm bench`/`LLM()` died at warmup with:

```
RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
```

FlashInfer JIT-compiles its top-k/top-p sampling kernel at first use and needs `nvcc`. This box has
the driver (595.84) and the CUDA runtime libs torch bundles, but no CUDA **toolkit**.

`~/research/FoT/model_serving/serve_vllm.sh` already hit this in July and worked around it:

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0    # line 32
```

...and the comment below it (lines 34–37) notes a **second** casualty: DeepGEMM's FP8 JIT path also
needs `nvcc`, so vLLM silently falls back to `CutlassFp8BlockScaledMMKernel`.

That second one is the interesting one. **Every FP8 number produced on this box so far was measured
on a fallback GEMM kernel.**

### Why only the sampler crashes (verified 2026-08-01)

Two different failure-detection designs, which is why `VLLM_USE_FLASHINFER_SAMPLER=0` is needed but
`VLLM_USE_DEEP_GEMM=0` is not:

- **DeepGEMM — probe before use.** Gated behind `has_deep_gemm()`
  (`vllm/utils/import_utils.py:475`), a module-presence check. The vendored copy at
  `vllm/third_party/deep_gemm/__init__.py:108-117` runs `which nvcc`, falls back to
  `/usr/local/cuda`, then `assert cuda_home is not None` — a bare assert that fires at **import
  time**. Caught, `has_deep_gemm()` → `False`, silent Cutlass fallback. Confirmed live.
- **FlashInfer sampler — assume, then JIT lazily.** `flashinfer_sampler_supported()` checks env var
  + platform + compute capability only; its docstring says it *assumes* flashinfer is installed.
  `flashinfer-python` imports fine (Python wrapper). `nvcc` isn't needed until the first kernel
  call, inside warmup, unguarded → fatal.

**Premise confirmed:** the only thing blocking DeepGEMM is `which nvcc` / `/usr/local/cuda`. Not a
missing package, not sm_120 support. A real toolkit install flips it on — so the A/B is viable.

### The infra task

Install the CUDA toolkit properly — system-wide at `/usr/local/cuda`, from NVIDIA's apt repo,
matching the 13.x series torch is built against (currently `torch 2.11.0+cu130`, `nvcc` 13.3 is
already present as a pip dep but incomplete: no CUB headers, undiscoverable by FlashInfer).

Needs sudo and a few GB. Also a prerequisite for ever building vLLM from source.

*Not* a hack to work around — the pip `nvidia/cu13` tree + `CUDA_HOME` pointed into a venv was
considered and rejected as fragile.

**This is the week's one external-latency item — do it first**, before designing anything. If the
toolkit install fails or `has_deep_gemm()` still returns `False`, there is no post and the week
should pivot immediately rather than at day three.

### The question

To be written down before the A/B runs and left unedited. The finding is whichever way it comes out.

**Question.** Every FP8 number measured on this box so far ran on `CutlassFp8BlockScaledMMKernel`,
because a missing CUDA toolkit silently disabled DeepGEMM's JIT path. Install the toolkit and the
DeepGEMM path becomes available. **How much does the kernel choice actually change serving
performance on sm_120?**

**Why it's worth asking.** The interesting part isn't the delta, it's that the fallback was
*silent* — a presence probe failed at import and vLLM carried on. So the question underneath is:
how much can a benchmark be wrong by, with nothing in the logs to tell you?

**Predictions, to be checked against the result.**

1. The sampler arm (`VLLM_USE_FLASHINFER_SAMPLER`) moves throughput by ≈0. Sampling is not on the
   critical path at these batch sizes.
2. The DeepGEMM arm shows a delta that **grows with batch size** — decode at low concurrency is
   memory-bandwidth-bound, so GEMM choice is largely hidden; prefill and high-concurrency decode are
   compute-bound, where it should show.
3. Direction unknown in magnitude. Cutlass FP8 on Blackwell may simply be well-tuned, in which case
   the delta is small at every batch size.

**What each outcome means.**

| Result | The post |
|---|---|
| Delta grows with batch size, materially | The number, plus the mechanism: a silent fallback cost me X% and nothing warned me |
| Delta flat and small at all batch sizes | "Cutlass FP8 on Blackwell is already good" — a null result worth publishing, because the *reason* I went looking is the transferable part |
| `has_deep_gemm()` still `False` after install | No post. `LOG.md`, and pivot the week |

Prediction 3 is the honest one: **there is no assumption here that the card was being
under-served.** If the delta is negligible, the correction is to the *inference* I drew on
2026-08-01 ("my numbers are probably understated"), not to the numbers themselves — and saying so
is the post.

Whichever way it lands, the framing continues week 1's voice: the mechanism of being wrong is more
transferable than the measurement. It should not be a second self-flagellation post for its own
sake — if the honest answer is "I was wrong to worry," say that plainly and move on.

### Design

**The A/B is env-var toggled, not install/uninstall.** Install the toolkit once, then flip switches.
Verified present in vLLM 0.26.0 `envs.py`:

| Switch | Controls |
|---|---|
| `VLLM_USE_DEEP_GEMM` | FP8 block-scaled GEMM — the one expected to matter |
| `VLLM_USE_FLASHINFER_SAMPLER` | top-k/top-p sampling kernel — expected ≈ 0 |
| `VLLM_MOE_USE_DEEP_GEMM` | MoE path; only if an MoE model enters the mix |
| `VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER` | alternative FP8 GEMM backend — a possible third arm |

**Measure the two effects separately.** Reporting a combined "toolkit on vs off" delta and letting
readers assume it's all one thing is exactly the sloppiness this post is supposed to be correcting.
Three arms minimum: both off (the July configuration), sampler only, DeepGEMM only.

**Model:** `Qwen/Qwen3-32B-FP8` — already in the HF cache, and FP8 is what exercises DeepGEMM.
A BF16 arm is not needed; this is a kernel comparison, not a quantization comparison.

**Hypothesis to test, not to assume:** the DeepGEMM delta should *grow with batch size* — decode at
low concurrency is memory-bandwidth-bound, so GEMM kernel choice is largely hidden; prefill and
high-concurrency decode are compute-bound, where it should show. If that shape appears, the chart
is delta-vs-concurrency and the post writes itself. **If the delta is flat and small, there is no
post** — say so in `LOG.md` and drop it. Cutlass FP8 on Blackwell may simply be good.

**Run it with `vllm bench sweep serve`.** The experiment is arms × concurrency levels, which is a
Cartesian product over server env-vars × bench params — precisely what the tool takes. Env vars
aren't `vllm serve` flags, so the arms go in via `--serve-cmd` wrappers (one per arm) or an
`--after-bench-cmd`/env-injection approach; **resolve this with `--dry-run` before committing to an
unattended run.** `--num-runs 3` matters here: if the DeepGEMM delta is small, single-shot runs can't
distinguish it from noise, and "small but real" vs "noise" is the entire finding.

**Carry week 1's settled configuration forward** so the two weeks are comparable where they overlap:
`--no-enable-prefix-caching`, `--ignore-eos`, pinned `max_num_seqs` / `max_num_batched_tokens`,
`--save-result --save-detailed --metadata`, and no `--reasoning-parser`.

### Confounds specific to this one

- **Do not compare against the July numbers.** Those came from a different vLLM, different torch,
  possibly different driver. Both arms get re-measured today, same binary, same session, same
  everything but the env var. This is the whole ballgame — a stale-baseline comparison would make
  the correction post itself wrong.
- **Confirm the kernel actually swapped.** `VLLM_USE_DEEP_GEMM=1` with a broken toolkit falls back
  silently and produces a null result that looks like a finding. Cheap precise probe, before any
  benchmarking:

  ```bash
  python -c "from vllm.utils.import_utils import has_deep_gemm; print(has_deep_gemm())"
  ```

  Must print `True` after the toolkit install. Also grep each run's startup log for
  `CutlassFp8BlockScaledMMKernel` and record which kernel that arm actually ran. Non-negotiable.
- **JIT compile time is not inference time.** First run after enabling pays a multi-minute compile
  (cached in `~/.cache/flashinfer`). Warm up, discard, then measure. Note `sweep serve` restarts the
  server per serve-config — make sure the compile happens inside a warmup combo, not inside a
  measured one.
- Plus the standing list in `CLAUDE.md` — APC off, `ignore_eos`, clock drift across a longer sweep.
  A sweep with `--num-runs 3` is *much* longer than week 1's seven runs, so clock drift goes from a
  minor check to a real risk. Record clocks per run and check whether arm order correlates with
  temperature; if it does, interleave arms rather than running them in blocks.

### Open questions

- Is `VLLM_HAS_FLASHINFER_CUBIN` a prebuilt-kernel path that sidesteps `nvcc` entirely? If so, the
  "you need a toolkit" framing may be too strong and the post needs rewording.
- Does the FP8 delta show up at all on sm_120, or only on datacenter Blackwell?

---

## Other candidates (parked, not competing for week 2)

Recorded so they aren't rediscovered. None of these is planned.

- **What does prefix caching actually buy?** Week 1 disabled APC as a confound; the inverse is the
  experiment. `--dataset-name prefix_repetition` is purpose-built
  (`--prefix-repetition-num-prefixes`, `--prefix-repetition-prefix-len`,
  `--prefix-repetition-suffix-len`), and `/reset_prefix_cache` gives a clean per-run reset. Natural
  sequel to week 1 and cheap.
- **The one flag that moves TTFT most.** Week 1 deliberately shipped untuned defaults
  (`max_num_batched_tokens=2048`, `max_num_seqs=128`). Sweeping them against the week-1 baseline is a
  direct follow-up with a built-in hook: "I left the defaults on purpose. Here's what they cost."
- **Speculative decoding with a local target.** `Qwen/Qwen3.5-9B` (already cached) has
  `mtp_num_hidden_layers = 1` — a built-in multi-token-prediction draft head. That satisfies
  `CLAUDE.md`'s hard constraint (target must have published draft/EAGLE weights) with a model already
  on disk. Note it's also a hybrid linear-attention VLM, so acceptance-rate results won't transfer to
  dense models — scope any claim accordingly.
- **Arrival realism.** Week 1's sweep is synthetic Poisson (`--burstiness 1.0`). `--dataset-name
  burstgpt` replays real production traces, and `sweep serve_workload` exists for workload-shaped
  sweeps. Answers "but real traffic isn't Poisson."

---

## Results

<!-- fill in as runs complete -->

## Notes / what surprised me

<!-- -->
