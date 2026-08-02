# Week 2 — CANDIDATE, not yet planned

> Captured mid-week-1 (2026-08-01) while debugging env setup. Per the cadence rule in `CLAUDE.md`,
> week 2 gets planned at the *start* of week 2, from what week 1 actually produced. This file is a
> holding pen for the idea and its design, not a commitment.

---

## Carried in from week 1

- **Sweep harness.** Week 1 explicitly deferred it and ran seven benchmarks by hand.

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

### The post idea

**"I understated my own hardware."** Self-correction as the hook: I benchmarked this card in a
hurry, bypassed two kernel paths because the toolkit wasn't installed, and published numbers that
were too low. Here's what the card actually does, and here's how much the shortcut cost.

Stronger than a plain benchmark post for the same reason week 1's post 5 is: the admission travels
further than the curve. It also pairs naturally with the week-1 confound-confession framing, so the
two posts build a consistent voice rather than repeating a trick.

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

**This is where the deferred harness earns itself.** The experiment is arms × concurrency levels —
a sweep, run unattended. Building the harness *for* this is better than building it in the
abstract, and it collapses the two week-2 candidates into one.

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
  (cached in `~/.cache/flashinfer`). Warm up, discard, then measure.
- Plus the standing list in `CLAUDE.md` — APC off, `ignore_eos`, clock drift across a longer sweep.

### Open questions

- Is `VLLM_HAS_FLASHINFER_CUBIN` a prebuilt-kernel path that sidesteps `nvcc` entirely? If so, the
  "you need a toolkit" framing may be too strong and the post needs rewording.
- Does the FP8 delta show up at all on sm_120, or only on datacenter Blackwell?

---

## Results

<!-- fill in as runs complete -->

## Notes / what surprised me

<!-- -->
