# TPOT vs ITL — same measurement, two aggregations

Resolved 2026-08-08 while reading the warmup output, where `median ITL (15.09) < median TPOT (17.36)`
looked like a contradiction. It isn't.

## Both exclude the first token

The initial suspicion — that ITL includes the first token and so must be ≥ TPOT — is wrong.
`benchmarks/lib/endpoint_request_func.py:233-242`:

```python
if not first_chunk_received:
    first_chunk_received = True
    output.ttft = time.perf_counter() - st          # first token → TTFT only, NOT appended to itl
else:
    output.itl.append(timestamp - most_recent_timestamp)   # every later token → itl
```

TPOT is the same scope (`benchmarks/serve.py:610`):

```python
tpot = latency_minus_ttft / (output_len - 1)
```

Since `latency − ttft` is the sum of that request's gaps and `output_len − 1` is their count:

```
tpot_i  ==  mean(request i's ITL gaps)      exactly
```

**They measure the identical set of intervals.** The only difference is when the averaging happens.

## Exact computation

Given per-request gap arrays `gaps = {req_1: [...], ..., req_N: [...]}`:

```python
# TPOT — collapse each request FIRST, then aggregate over N values
tpots = [sum(gaps[i]) / len(gaps[i]) for i in range(N)]
mean_tpot, median_tpot, p99_tpot = np.mean(tpots), np.median(tpots), np.percentile(tpots, 99)

# ITL — concatenate everything, then aggregate over sum(len) values
itls = [g for i in range(N) for g in gaps[i]]
mean_itl, median_itl, p99_itl = np.mean(itls), np.median(itls), np.percentile(itls, 99)
```

| | population | size in the 32-request warmup |
|---|---|---|
| TPOT | one mean **per request** | 32 |
| ITL | every gap, **pooled** | 32 × 255 = 8160 |

## What is and isn't guaranteed

**Means are equal** when all output lengths match:

```
mean_itl  = (1/(N·M)) ΣᵢΣⱼ gᵢⱼ
mean_tpot = (1/N) Σᵢ [(1/M) Σⱼ gᵢⱼ]  =  (1/(N·M)) ΣᵢΣⱼ gᵢⱼ
```

Warmup: `18.34 == 18.34`. That identity is a **free check that `--ignore-eos` worked** — with unequal
output lengths it breaks, because `mean_tpot` weights each *request* equally while `mean_itl` weights
each *token* equally.

**Medians and percentiles have no ordering guarantee.** They're medians of different distributions.

Observed direction, and why it's expected: the gap distribution is **right-skewed** — most gaps sit
near the clean decode step time, a minority are large (prefill chunks stealing the step). Averaging
*within* a request pulls each request's mean above its own median, so the N request-means sit to the
right of the raw-gap distribution:

```
median_tpot (17.36)  >  median_itl (15.09)     ← averaging lifts each request above its own mode
p99_itl (95.05)     >>  p99_tpot (27.95)       ← averaging dilutes any single stall; a request-mean
                                                 can never reach the raw tail
```

So `P99 ITL` sees an individual decode step stalling behind a prefill chunk. `P99 TPOT` structurally
cannot.

## Percentile resolution — the trap

`p99_tpot` over 32 requests is meaningless. `np.percentile` with linear interpolation places the 99th
percentile at index `0.99 × 31 = 30.69`, i.e. interpolating between the two largest values. **P99 TPOT
from a 32-request run is "the worst request's average," not a percentile.** `p99_itl` over 8160
samples is properly resolved.

Consequence: **don't quote P99 TPOT from small runs.** This is a second reason the `--num-prompts`
scaling in the run table matters, beyond steady-state duration. Worth checking how P99 TPOT
stabilises between run 1 (200 prompts) and run 6 (1500).

## Which to use for what

| purpose | statistic | why |
|---|---|---|
| the SLO (`--goodput tpot:50`) | **per-request TPOT** | the SLO is a per-user promise; a user experiences their own average |
| calibrating `c` in [`napkin-model.md`](napkin-model.md) | **median ITL** | pooling exposes the mode — the clean, uncontaminated decode step |
| diagnosing prefill interference | **ITL percentiles** | the pooled tail shows stalls that request-averaging hides |
| comparing runs with different output lengths | **TPOT** | token-weighting makes `mean_itl` shift with length mix |

## For the post

Post 4 material: the same 8160 measurements, aggregated two ways, and only one aggregation can see
the stall. `median ITL` is the machine's clean step time; `P99 ITL` is chunked prefill preempting
decode; `TPOT` is what a user actually feels. Quoting all three and saying why they differ signals
having read the tool rather than pasted its output.

One implementation detail: vLLM appends to `tpots` only when `output_len > 1` (`serve.py:607-611`),
so `len(tpots)` can be below the request count in runs with short generations. Irrelevant under
`--ignore-eos` at a fixed 256.
