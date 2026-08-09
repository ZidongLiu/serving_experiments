# Week 2 — a cost model that predicts models it has never run

**Goal.** Turn week 1's [napkin model](../week1/notes/napkin-model.md) into a formula grounded in the
vLLM V1 scheduler, then use it to **predict, commit, push, and only then measure** — on models never
benchmarked on this card. The deliverable is the diff between the committed predictions and what the
card actually did.

The repo is public, so the git timestamp is third-party proof the predictions came first. That
mechanic *is* the post: anyone can publish a benchmark, almost nobody publishes a number before
they know it.

---

## Still to work out

Open, in rough order of what blocks what:

- **Which scalars to predict.** Means are predictable; goodput is a threshold on a *distribution*
  and probably isn't, yet. Candidate targets: median ITL at low load, mean TTFT at low load, the
  capacity wall in req/s, and the rate at which goodput peaks.
- **Which models, and what each one isolates.** Contrasts rather than a survey — one pair that
  changes only weight bytes while holding architecture fixed would test the load-bearing term
  directly.
- **How much of the scheduler the formula actually needs.** Read
  `vllm/v1/core/sched/scheduler.py` end to end first; the answer probably falls out of the token
  budget split between running and waiting requests.
- **The FP8 confound.** No CUDA toolkit here, so FP8 GEMM runs a Cutlass fallback. That moves the
  compute term, which is fine for bandwidth-bound decode claims and not fine for prefill/TTFT ones.
  See [`../candidates.md`](../candidates.md).
- **Run budget.** Testing the formula needs floors and the wall, not full sweeps — so far fewer runs
  per model than week 1 used.
- **What means no post.** If every model is purely bandwidth-bound, the formula collapses to one term
  and "it works" is a tautology. Needs a stated falsification bar before any measuring.

## Predictions

<!-- Written and pushed BEFORE running anything. Left unedited afterwards. -->

## Results

<!-- filled in as runs complete -->

## Notes / what surprised me

<!-- raw, unedited daily notes -->
