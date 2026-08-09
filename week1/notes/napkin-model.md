# Napkin model — where time goes in one forward pass

A first-principles cost model for this box + this model, used to **predict every run before
measuring it**. The point is not accuracy. The point is that a prediction from a model is wrong in a
*specific way*, so the residual tells you which term you misunderstood. A random guess, when wrong,
teaches nothing.

Status: **model stated, constants not yet calibrated.** Fill in §4 and §5 by hand.

---

## 1. Symbols

| symbol | meaning | source |
|---|---|---|
| `P` | parameters | model card |
| `W` | weight bytes = `P × bytes_per_param` | config (`dtype`) |
| `k` | KV bytes per token per sequence = `2 × layers × kv_heads × head_dim × bytes_per_elem` | config |
| `BW` | memory bandwidth | spec sheet |
| `C` | **effective** compute throughput (FLOP/s) | **must measure** |
| `c` | fixed per-step overhead (launch, sampling, Python) | **must measure** |
| `B` | sequences currently **decoding** | `vllm:num_requests_running` |
| `B_p` | requests with **prefill** work in a step | derived from `T`, `S` |
| `L` | context length of a decoding sequence | `S` … `S+O` |
| `S`, `O` | input / output sequence length | benchmark flags |
| `T` | `max_num_batched_tokens` — token budget per scheduler step | server flag |

Keep `B` and `B_p` distinct. A sequence being prefilled is **not** decoding yet; it joins `B` when it
emits its first token. They are coupled through `T`, never multiplied together.

## 2. Two phases, two bottlenecks

Everything below follows from one asymmetry:

- **Prefill** processes many tokens against one set of weights → high arithmetic intensity →
  **compute-bound**. Measure it in FLOPs.
- **Decode** processes one token per sequence, re-reading every weight to do almost no math →
  **bandwidth-bound**. Measure it in bytes.

Counting decode in FLOPs, or prefill in bytes, tells you nothing. Keeping them in separate units is
most of what makes the model predictive.

## 3. The equations

### Decode

```
t_decode  =  W/BW               (read all weights — INDEPENDENT of B)
          +  (B × L × k)/BW     (re-read all KV for all decoding sequences)
          +  c                  (fixed overhead)

TPOT      =  t_decode           (when no prefill is interleaved in the step)
```

`W/BW` is a **floor**: paid whether one sequence decodes or a hundred. That is why batching is nearly
free for throughput, and why per-GPU throughput rises ~linearly in `B` until the KV term or compute
takes over.

The KV term grows with `B × L`, so it re-reads history that gets longer every step. Within one
request, `L` runs from `S` to `S+O`, so the **average over the decode phase is `(2S + O)/2`** — and
ITL should drift slightly upward across a request's own output.

### Prefill

```
FLOPs(n)  =  2 × P × n                    (linear layers — linear in n)
          +  a × layers × n² × d_model    (attention — quadratic in n)

t_prefill(n) = FLOPs(n) / C
```

`n` is a count of **new** tokens being computed. It takes different values depending on the question:

| question | `n` |
|---|---|
| cost of one request's whole prefill | `S` |
| cost of one scheduler step | `min(prefill work remaining, T − B)` |

The `n²` term is **per-sequence** — attention never crosses request boundaries, so for a step packing
chunks from several requests, sum the per-request terms rather than squaring the packed total.

**Compute the crossover** where `a·layers·n²·d` overtakes `2·P·n`. That single number governs all
long-context reasoning and is the reason KV/long-context is a separate week.

### The scheduler couples them — this is what produces the goodput curve

Confirmed in `vllm/v1/core/sched/scheduler.py`: `token_budget = max_num_scheduled_tokens` (line 445),
the **running** loop consumes it first (471–619), and the **waiting** loop gets only what's left
(669–1029).

```
n_prefill_per_step  =  T − B          (decodes are served FIRST; prefill gets the remainder)

t_step  ≈  t_prefill(n_prefill_in_step)  +  t_decode
```

So step duration is **bimodal**: a pure-decode step costs `t_decode`; a step dragging a full prefill
chunk costs far more. That one fact explains any large gap between median ITL and P99 ITL.

### TTFT

```
TTFT  =  queue_wait  +  t_prefill(S)

queue_wait  ≈  (Q / n_prefill_per_step) × t_step        Q = prefill tokens queued ahead
```

Equivalently, and easier to use:

```
TTFT  ≈  (all prefill tokens that must be processed before yours completes) / prefill_token_rate
```

**Read the coupling.** As offered load rises: `B` grows → `n_prefill_per_step = T − B` *shrinks* →
`Q` grows. Both effects multiply, so **TTFT degrades superlinearly in load while throughput merely
asymptotes.** That divergence is week 1's post, derived rather than observed.

### From total work to the reported metric

Total prefill work is linear in the number of requests prefilling — but that does **not** multiply
your own `t_prefill(S)`. It lands in the queue. For `B_p` requests of length `S` arriving together
and served in order:

```
TTFT(first served)  ≈  1   × t_prefill(S)
TTFT(last served)   ≈  B_p × t_prefill(S)
mean TTFT           ≈  ((B_p + 1)/2) × t_prefill(S)
```

`T` slices that work across `⌈B_p·S / T⌉` steps. Slicing changes *granularity*, not total: same
FLOPs, same total time. Note it is `T` that binds here, **not `max_num_seqs`** — `max_num_seqs` caps
sequences, `T` caps tokens per step, and with long prompts `T` binds first.

Known deviations from this floor, in order of size:

1. **Decode steals budget.** Once early requests start generating they claim `B` tokens per step,
   shrinking `T − B`, so late-served requests degrade *worse* than linearly.
2. **Steppy distribution.** Requests complete in groups of `⌊T/S⌋`, so TTFT clusters rather than
   spreading evenly. A uniform measured TTFT distribution falsifies this assumption.

## 4. Constants — TO FILL IN BY HAND

Two come free from config, one from the spec sheet, two must be earned from measurements.

| constant | value | how obtained |
|---|---|---|
| `P` | | model card |
| `W` | | `P × bytes_per_param` |
| `k` | | `2 × layers × kv_heads × head_dim × bytes` |
| `BW` | | spec sheet |
| `C` | | **invert:** `C = FLOPs(tokens_prefilled) / time_spent_prefilling` |
| `c` | | **invert:** `c = t_decode_measured − W/BW − (B·L·k)/BW` |

Calibration sources available now, before run 1:

- `C` ← from `../upload_result/warmup_bench.txt`: 32 requests × 1024 tokens prefilled, and P99 TTFT bounds
  when the last prefill completed.
- `c` ← from the same file: **median** ITL is the clean decode-only step (the mean is contaminated by
  prefill interleaving; use the median). Why median ITL specifically, and not median TPOT or mean
  ITL: [`tpot-vs-itl.md`](tpot-vs-itl.md).

Spec-sheet peak FLOPS is *not* `C`. Real MFU is a fraction of it, and using peak will make every
prefill prediction optimistic by 2–3×.

## 5. Predictions — write BEFORE running

One row per run, filled in *before* the command executes. The residual column is the whole point.

| run | rate | pred TTFT | pred TPOT | pred tok/s | measured | residual → which term was wrong? |
|---|---|---|---|---|---|---|
| 1 | 1 | | | | | |
| 2 | 2 | | | | | |
| 3 | 4 | | | | | |
| 4 | 8 | | | | | |
| 5 | 16 | | | | | |
| 6 | 32 | | | | | |
| 7 | inf | | | | | |

Run 1 is special: `B ≈ 1` and `Q ≈ 0`, so `TTFT ≈ t_prefill(S)` — a single term. It is the cleanest
possible measurement of `C`, and it also decides whether `ttft:200` is a viable threshold or too
tight to produce a curve that climbs before it falls.

## 6. Falsification checks

A model is only useful if it can be wrong in a specific way. Three places this one should hold:

1. `t_decode` predicted vs **median** ITL, at known `B` and `L`.
2. `t_prefill(T − B)` predicted vs the step duration backed out of `TTFT_p99 ÷ number of chunks`.
3. **TPOT roughly flat as `B` grows at fixed `L`** — because `W/BW` does not depend on `B`.

If (3) fails, the bandwidth-bound assumption has broken and something else is binding. That is a
finding, not an error — write it down.

## 7. Why bother

The FLOPs-based prefill term is what lets you predict performance on hardware you **don't have** —
the rented multi-GPU work in `CLAUDE.md`'s direction section. Empirical calibration only works on the
box in front of you; the model transfers.

Reading queued behind this, for a later week: `vllm/v1/core/sched/scheduler.py` end to end (~90 min —
it is literally the code that produces every number in the sweep), Kipply's *Transformer Inference
Arithmetic* for the roofline, the DistServe paper for the goodput framing.
