#!/usr/bin/env python3
"""Week 1 chart: throughput plateaus while goodput peaks and collapses.

Reads `upload_result/summary.json` — the committed, slimmed output of slim.py — so
the chart reproduces from a fresh clone without the 49 MB of raw benchmark JSONs.
The chart is written back to upload_result/, since it is a published artifact.

Goodput is recomputed here rather than read from the stored value, because the whole
point is that goodput depends on the SLO you choose. The summary keeps per-request
TTFT and TPOT, which is exactly what a threshold can see, so any SLO is available
offline without re-running anything.

The strict tier is recomputed with the same thresholds vLLM used at run time, so its
agreement with the stored `request_goodput` is a self-check on two things at once:
this script's arithmetic, and whether slim.py lost any fidelity. If it fails, the
loose tier is not trustworthy either.

Usage:  uv run python week1/src/slim.py && uv run python week1/src/plot.py
"""

from __future__ import annotations

import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
WEEK_DIR = os.path.dirname(HERE)
UPLOAD_DIR = os.path.join(WEEK_DIR, "upload_result")
SUMMARY = os.path.join(UPLOAD_DIR, "summary.json")
OUT_PNG = os.path.join(UPLOAD_DIR, "week1_goodput.png")

# (label, ttft_ms, tpot_ms)
STRICT = ("strict SLO (TTFT 200ms / TPOT 50ms)", 200.0, 50.0)
LOOSE = ("loose SLO (TTFT 1000ms / TPOT 100ms)", 1000.0, 100.0)

# Highest rate drawn. Everything above is the same saturated state, so plotting it
# only steals x-axis room from the knee. Table still reports every run.
X_MAX = 16.0


def rate_of(d: dict) -> float:
    """Request rate as a float; `inf` for the saturation run (stored as "inf")."""
    try:
        v = float(d.get("request_rate"))
    except (TypeError, ValueError):
        return math.inf
    return math.inf if math.isnan(v) else v


def goodput(d: dict, ttft_bound_ms: float, tpot_bound_ms: float) -> float:
    """Requests/s meeting BOTH thresholds, recomputed from per-request latencies.

    Mirrors vLLM's definition. `tpot_ms` is None for requests with a single output
    token (no inter-token gaps to average); vLLM skips those, so this does too.
    """
    pr = d["per_request"]
    good = sum(
        1
        for ttft, tpot in zip(pr["ttft_ms"], pr["tpot_ms"])
        if tpot is not None and ttft <= ttft_bound_ms and tpot <= tpot_bound_ms
    )
    return good / d["duration"]


def main() -> None:
    if not os.path.exists(SUMMARY):
        raise SystemExit(f"{SUMMARY} not found — run: uv run python week1/src/slim.py")

    runs = sorted(json.load(open(SUMMARY))["runs"], key=rate_of)

    print(f"{'rate':>5} {'tok/s':>8} {'strict':>8} {'stored':>8} {'delta':>7} "
          f"{'loose':>8} {'TTFT_mean':>10} {'TTFT_p99':>10}")
    print("-" * 76)

    rates, thru, gp_strict, gp_loose, ttft_mean, ttft_p99 = [], [], [], [], [], []
    sat_thru = None
    worst_delta = 0.0

    for d in runs:
        r = rate_of(d)
        gs = goodput(d, STRICT[1], STRICT[2])
        gl = goodput(d, LOOSE[1], LOOSE[2])
        stored = d.get("request_goodput", float("nan"))
        delta = abs(gs - stored)
        worst_delta = max(worst_delta, delta)

        label = "inf" if math.isinf(r) else f"{r:g}"
        print(f"{label:>5} {d['output_throughput']:>8.1f} {gs:>8.2f} {stored:>8.2f} "
              f"{delta:>7.4f} {gl:>8.2f} {d['mean_ttft_ms']:>10.1f} {d['p99_ttft_ms']:>10.1f}")

        if math.isinf(r):
            # Can't sit on a numeric x-axis; drawn as a reference line instead.
            sat_thru = d["output_throughput"]
            continue
        if r > X_MAX:
            # Rates above the wall are all the same saturated state (see the table).
            # Plotting them would spend a third of the x-axis on a flat line and
            # squeeze the 8-16 knee, which is the part with the finding in it.
            continue

        rates.append(r)
        thru.append(d["output_throughput"])
        gp_strict.append(gs)
        gp_loose.append(gl)
        ttft_mean.append(d["mean_ttft_ms"])
        ttft_p99.append(d["p99_ttft_ms"])

    print("-" * 76)
    print(f"self-check: worst |recomputed strict - stored request_goodput| = {worst_delta:.5f} req/s")
    if worst_delta > 0.01:
        print("  WARNING: recomputation disagrees with vLLM — do not trust the loose tier")

    peak_i = max(range(len(gp_strict)), key=lambda i: gp_strict[i])
    print(f"peak strict goodput: {gp_strict[peak_i]:.2f} req/s at rate {rates[peak_i]:g} "
          f"(throughput there: {thru[peak_i]:.0f} tok/s)")
    if sat_thru:
        print(f"saturation throughput: {sat_thru:.0f} tok/s "
              f"({100 * thru[peak_i] / sat_thru:.0f}% of it reached at the goodput peak)")

    # ── chart ────────────────────────────────────────────────────────────────
    fig, (ax_t, ax_l) = plt.subplots(
        2, 1, figsize=(9, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.12},
    )

    # Panel A — throughput vs goodput. Goodput is converted to tok/s (x256, exact
    # because --ignore-eos fixed every output at 256 tokens) so both live on ONE
    # axis: the gap between the curves is literally wasted work.
    OSL = 256
    ax_t.plot(rates, thru, "o-", lw=2, ms=6, color="#2a78d6",
              label="total output throughput")
    ax_t.plot(rates, [g * OSL for g in gp_loose], "s-", lw=2, ms=6, color="#1baf7a",
              label=f"useful throughput — {LOOSE[0]}")
    ax_t.plot(rates, [g * OSL for g in gp_strict], "^-", lw=2, ms=6, color="#eb6834",
              label=f"useful throughput — {STRICT[0]}")

    if sat_thru:
        ax_t.axhline(sat_thru, color="#8a8a85", ls=":", lw=1.5)
        ax_t.annotate(f"saturation (rate=∞): {sat_thru:.0f} tok/s",
                      xy=(rates[0], sat_thru), xytext=(4, -12),
                      textcoords="offset points", ha="left", va="top",
                      fontsize=9, color="#52514e")

    ax_t.annotate(
        f"goodput peaks here\n{gp_strict[peak_i]:.2f} req/s at {rates[peak_i]:g} req/s offered\n"
        f"{thru[peak_i]:.0f} tok/s = {100 * thru[peak_i] / (sat_thru or thru[-1]):.0f}% of peak",
        xy=(rates[peak_i], gp_strict[peak_i] * OSL),
        xytext=(rates[peak_i] + 3.5, gp_strict[peak_i] * OSL + 250),
        fontsize=9, color="#0b0b0b",
        arrowprops=dict(arrowstyle="->", color="#52514e", lw=1.2),
    )

    ax_t.set_ylabel("tokens / s")
    ax_t.set_title(
        "Qwen3-8B on 1× RTX PRO 6000 Blackwell — throughput plateaus, useful work collapses\n"
        "vLLM 0.26.0, 1024-in/256-out, prefix caching off, untuned defaults "
        "(max_num_seqs=128, max_num_batched_tokens=2048)",
        fontsize=10.5, loc="left", pad=12,
    )
    ax_t.legend(frameon=False, fontsize=9, loc="lower left")
    ax_t.grid(alpha=0.25, lw=0.6)
    ax_t.set_ylim(bottom=0, top=(sat_thru or max(thru)) * 1.12)
    ax_t.set_xlim(0.3, X_MAX + 0.7)

    # Panel B — why: TTFT blows through the thresholds past the capacity wall.
    ax_l.plot(rates, ttft_mean, "o-", lw=2, ms=6, color="#e34948", label="mean TTFT")
    ax_l.plot(rates, ttft_p99, "o--", lw=2, ms=6, color="#e34948", alpha=0.6,
              label="P99 TTFT")
    for thresh, name in ((STRICT[1], "strict TTFT SLO"), (LOOSE[1], "loose TTFT SLO")):
        ax_l.axhline(thresh, color="#8a8a85", ls=":", lw=1.5)
        ax_l.annotate(f"{name} ({thresh:.0f} ms)", xy=(X_MAX + 0.5, thresh),
                      xytext=(-2, 5), textcoords="offset points",
                      ha="right", fontsize=8.5, color="#52514e")

    ax_l.set_yscale("log")
    ax_l.set_ylabel("TTFT (ms, log scale)")
    ax_l.set_xlabel("offered request rate (req/s)")
    ax_l.legend(frameon=False, fontsize=9, loc="upper left")
    ax_l.grid(alpha=0.25, lw=0.6, which="both")
    ax_l.set_xticks(rates)
    ax_l.set_xticklabels([f"{r:g}" for r in rates])

    fig.text(0.005, 0.005,
             f"Single runs, no repeats. Rates above {X_MAX:g} omitted from the chart — identical "
             "saturated behaviour. Above the ~9 req/s capacity wall there is no steady state: "
             "mean TTFT grows with run length, so those values describe (server, num_prompts).",
             fontsize=8, color="#52514e")

    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nwrote {OUT_PNG}")


if __name__ == "__main__":
    main()
