#!/usr/bin/env python3
"""Condense the raw `vllm bench serve` result JSONs into one committable summary.

The raw files are ~48 MB, 80% of it the `itls` arrays (one list of 255 inter-token
gaps per request) and another 19% `generated_texts` — which is garbage here, since
the prompts are random token IDs.

But `itls` is what makes the result *verifiable*: goodput is a threshold applied to
per-request latencies, so anyone holding those latencies can recompute it at their
own SLO instead of taking the published number on faith. Dropping them entirely
would leave the chart unreproducible.

So keep exactly what any SLO recomputation needs — per-request TTFT and TPOT — and
throw away everything else. `tpot_i` is the mean of request i's inter-token gaps,
which collapses 255 floats into 1 without losing anything a goodput threshold can
see. ~48 MB becomes a few hundred KB.

This is the promotion step: `results/` holds raw machine output and is gitignored,
`upload_result/` holds the curated artifacts and is committed. Run after benchmark.sh,
then commit whatever lands in upload_result/.

Usage:  uv run python week1/src/slim.py
"""

from __future__ import annotations

import glob
import json
import math
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
WEEK_DIR = os.path.dirname(HERE)
RESULT_DIR = os.path.join(WEEK_DIR, "results")        # raw, gitignored
UPLOAD_DIR = os.path.join(WEEK_DIR, "upload_result")  # curated, committed
OUT = os.path.join(UPLOAD_DIR, "summary.json")

# Small curated files copied through from the raw dir as-is. warmup_bench.txt is here
# because notes/napkin-model.md calibrates the cost model's two unknown constants from
# it — without it that step isn't reproducible by anyone but me.
PASSTHROUGH = ["clocks.csv", "warmup_bench.txt", "server_start.txt"]

# Scalars worth carrying over verbatim so the summary stands alone.
SCALARS = [
    "num_prompts", "duration", "completed", "failed",
    "request_throughput", "output_throughput", "total_token_throughput",
    "total_input_tokens", "total_output_tokens",
    "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
    "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms",
    "mean_itl_ms", "median_itl_ms", "p99_itl_ms",
    "request_goodput",
    # provenance, from --metadata
    "model", "vllm", "apc", "max_num_seqs", "mnbt", "seed", "rate",
]


def rate_key(d: dict) -> float:
    try:
        v = float(d.get("request_rate"))
    except (TypeError, ValueError):
        return math.inf
    return math.inf if math.isnan(v) else v


def main() -> None:
    paths = sorted(glob.glob(os.path.join(RESULT_DIR, "openai-*.json")))
    if not paths:
        raise SystemExit(f"no raw result JSONs in {RESULT_DIR} — run benchmark.sh first")

    raw = sorted((json.load(open(p)) for p in paths), key=rate_key)
    runs = []

    for d in raw:
        r = rate_key(d)
        # JSON has no Infinity literal, so the saturation run's rate is a string.
        out = {"request_rate": "inf" if math.isinf(r) else r}
        for k in SCALARS:
            if k in d:
                out[k] = d[k]

        # Per-request latencies, in ms. tpot_i == mean of that request's ITL gaps
        # == (latency - ttft) / (output_len - 1). Requests with a single output
        # token have no gaps; vLLM skips them for TPOT, so they get None here.
        ttft_ms, tpot_ms = [], []
        for ttft, itls in zip(d["ttfts"], d["itls"]):
            ttft_ms.append(round(ttft * 1000.0, 4))
            tpot_ms.append(round(sum(itls) / len(itls) * 1000.0, 4) if itls else None)
        out["per_request"] = {"ttft_ms": ttft_ms, "tpot_ms": tpot_ms}
        runs.append(out)

    doc = {
        "generated_by": "week1/src/slim.py",
        "source": "vllm bench serve --save-result --save-detailed",
        "note": (
            "Per-request TTFT and TPOT in milliseconds, sufficient to recompute goodput "
            "at any SLO: a request counts if ttft_ms <= your TTFT bound and tpot_ms <= "
            "your TPOT bound; goodput = count / duration. Raw JSONs (49 MB, including "
            "full per-token ITL arrays) are not committed; regenerate with "
            "week1/src/benchmark.sh."
        ),
        "runs": runs,
    }

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(doc, f, separators=(",", ":"))

    raw_mb = sum(os.path.getsize(p) for p in paths) / 1e6
    print(f"{len(runs)} runs, {sum(len(r['per_request']['ttft_ms']) for r in runs)} requests")
    print(f"raw {raw_mb:.1f} MB  ->  {os.path.getsize(OUT) / 1e6:.3f} MB  "
          f"({os.path.getsize(OUT) / (raw_mb * 1e6) * 100:.2f}%)")
    print(f"wrote {OUT}")

    for name in PASSTHROUGH:
        src = os.path.join(RESULT_DIR, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(UPLOAD_DIR, name))
            print(f"copied {name}")
        else:
            print(f"NOTE: {name} not in {RESULT_DIR} — not copied")


if __name__ == "__main__":
    main()
