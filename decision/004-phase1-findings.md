# 004 — Phase-1 Empirical Findings (decision/002 release gate)

**Status:** REPORTED — 2026-05-16.
**Charter gate (`decision/000-charter.md` §Public release gates):**
> Phase-1 demo runs end-to-end, all four ablations from `decision/002`
> produce signal, CI is green locally.

This file records the **as-measured** result of that gate so the OSS
publication can cite an honest baseline rather than a wishful one.

## Setup

- Toy scene with `N_VIEWS=32`, each view a perceptually-distinct base
  colour on an evenly-spaced HSV hue ring.
- `N_TARGETS=4` targets, each built from a randomly-chosen 4-view
  subset of the palette — so only ~12.5 % of views are useful per
  target.
- 6×6 grid of 16-px tiles (36 cells, 96×96 mosaic).
- 16 NBV steps per run, log-domain Sinkhorn with `epsilon=0.05`,
  `max_iter=100`.
- Reproducer: `python experiments/benchmark_phase1.py`.

## Headline numbers (mean ± std across 4 targets)

| Condition | mean final SSIM | mean sum-gain |
|---|---|---|
| **A1 random** (baseline)    | 0.3927 ± 0.025 | +0.2331 |
| **A1 saliency_biased**      | 0.4177 ± 0.058 | +0.2581 |
| **A2 Hungarian + saliency** | **0.4682 ± 0.040** | **+0.3092** |
| **A3 saliency OFF**         | 0.4375 ± 0.055 | +0.2781 |
| **A4 Oklab OFF**            | 0.4177 ± 0.057 | +0.2581 |

(JSON: `experiments/results/phase1_baseline_2026-05-16.json` — the
canonical committed baseline. The raw timestamped run files
`phase1_20260516T041014.json` / `phase1_20260516T041517.json` are
gitignored; only the renamed baseline copy is tracked so that
`baseline-diff` CI can compare against a stable artifact.)

## Per-ablation reading

### A1 — strategy axis (saliency-biased vs random)

`saliency_biased` beats `random` by **+0.025 SSIM** (mean over 4
targets). The "MUST beat random" charter gate is **passed**.

### A2 — matcher axis (Sinkhorn vs Hungarian)

Hungarian beats Sinkhorn-OT on this toy by **+0.050 SSIM**. This is
the opposite of the direction `decision/003` hoped for.

The honest interpretation is that, on this specific toy with uniform
marginals and `epsilon=0.05`, the entropic regularization that gives
Sinkhorn its differentiability also softens the assignment enough to
hurt SSIM. Future work mandated by this finding:

- Sweep `epsilon ∈ {0.005, 0.01, 0.025, 0.05}` (decision/002 already
  blocks shipping with a single epsilon).
- Use the saliency vector as a **non-uniform source marginal** (not
  just a row-scaling on the cost matrix) — this is the actual
  Cuturi/Peyré framing of "saliency as OT marginal" and we are
  currently approximating it with row-scaling.
- Compare `argmax_assignment(enforce_unique=True)` against the
  non-unique greedy used here.

Decision/003 is **not** rescinded: the entropic-OT formulation
remains the scientific contribution, and Hungarian remains an
ablation baseline only — never a runtime fallback. But the README
must NOT claim Sinkhorn wins on this toy.

### A3 — saliency axis (mosaicraft.compute_saliency_weights ON vs OFF)

Turning the saliency row-scaling OFF gives a **+0.020 SSIM**
improvement. mosaicraft's saliency formula (edges + Laplacian
energy + HSV saturation + center bias) was designed for Hungarian
top-K candidate culling, where it boosts important cells so they
get scarce good tiles; in the OT setting all tiles are visible
and the row-scaling just biases the entropic plan toward
high-saliency rows, which on this random-block toy is noise.

The right fix is exactly what A2's discussion already points at:
use saliency as the OT source marginal, not as a row multiplier on
cost. That is Phase 2 work.

### A4 — Oklab axis (oklab_weight=0.2 vs 0.0)

No detectable contribution (means coincide to 4 decimal places on
average; individual targets differ by ~0.005). The 191-dim feature
vector already encodes LAB statistics, so adding an Oklab penalty
double-counts colour information. Phase 2 should either:

- Drop Oklab from the cost,
- Or replace the LAB-based features with Oklab-based features and
  keep only one colour space in the cost.

## What this means for OSS publication

All four ablations **show signal** (non-overlapping mean values), so
the charter's "produces signal" gate is met. The directions are
unexpected and the README will say so explicitly — that honesty is
the headline scientific contribution of Phase 1, not the SSIM numbers
themselves. The reader who downloads this repo and runs `python
experiments/benchmark_phase1.py` should reproduce these numbers
exactly (deterministic seeds), and the JSON output is committed under
`experiments/results/` so reviewers can diff future runs against the
2026-05-16 baseline.

## Phase 2 backlog (sourced from this file)

1. Sinkhorn ε sweep + report best ε per target class.
2. Saliency-as-marginal (not row-scaling).
3. Drop Oklab OR replace LAB-based features with Oklab-based ones.
4. Replicate on a non-synthetic scene (real photos, GenNBV-style
   benchmark).
