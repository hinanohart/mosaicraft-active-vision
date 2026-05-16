# 006 — Phase-2 Findings (saliency-as-marginal + ε sweep, statistically tested)

**Status:** REPORTED — 2026-05-16.
**Setup:** `experiments/benchmark_phase2.py`, N=4 seeds × N=4 targets = 16 paired runs per condition, ε ∈ {0.01, 0.05, 0.1}, two cost framings (row-scale vs saliency-as-marginal) × two argmax modes (greedy vs `enforce_unique=True`). All Sinkhorn variants compared against the same Hungarian baseline by paired bootstrap (5000 resamples, 95% CI).
**Reproducer:** `python experiments/benchmark_phase2.py` (also smoke-able with `MOSAICRAFT_AV_BENCH_SMOKE=1`).

## Headline (the honest one)

**Hungarian still wins.** Every Sinkhorn variant we tried lost to the Hungarian baseline with a 95% bootstrap CI strictly below zero. The gap between the best Sinkhorn (saliency-as-marginal, ε=0.1, greedy argmax) and Hungarian is **Δ = −0.028 SSIM [−0.041, −0.015]** — not large, but reliably negative.

| Condition | mean final SSIM | Δ vs Hungarian (95% CI) | Beats Hungarian? |
|---|---|---|---|
| **hungarian** (baseline) | **0.5024 ± 0.037** | — | — |
| marginal_eps=0.1_unique=0 | 0.4747 ± 0.046 | −0.0277 [−0.0406, −0.0146] | no |
| marginal_eps=0.1_unique=1 | 0.4668 ± 0.043 | −0.0356 [−0.0446, −0.0260] | no |
| rowscale_eps=0.1_unique=0 | 0.4609 ± 0.048 | −0.0415 [−0.0527, −0.0294] | no |
| marginal_eps=0.05_unique=0 | 0.4413 ± 0.046 | −0.0610 [−0.0727, −0.0491] | no |
| rowscale_eps=0.05_unique=0 | 0.4328 ± 0.049 | −0.0696 [−0.0809, −0.0582] | no |
| rowscale_eps=0.01_unique=0 | 0.4186 ± 0.039 | −0.0838 [−0.0925, −0.0751] | no |
| marginal_eps=0.01_unique=0 | 0.3986 ± 0.044 | −0.1038 [−0.1166, −0.0912] | no |

(Full JSON: `experiments/results/phase2_baseline_2026-05-16.json`, schema_version=2.)

## What we believed entering Phase 2

`decision/004` recorded three hypotheses for why Hungarian had beaten Sinkhorn on the Phase-1 toy:

1. **H1 (ε):** ε=0.05 was a single point; with a different ε, Sinkhorn would catch up.
2. **H2 (saliency framing):** Saliency was applied as a row-scale, not as the OT source marginal that `decision/003` §"Algorithm sketch" actually specified. Switching framings would close the gap.
3. **H3 (uniqueness):** The greedy `argmax_assignment` may have been overlapping cell choices; `enforce_unique=True` should help.

Phase 2 falsifies all three:

- **H1 falsified.** ε=0.1 is the best Sinkhorn ε on this toy; tightening to ε=0.05 or ε=0.01 makes the gap *worse*, not better. Sinkhorn's empirical sweet spot here is the softest end of our grid, and even that loses to Hungarian.
- **H2 falsified.** Saliency-as-marginal beats row-scale by a hair at ε=0.1 (0.4747 vs 0.4609), but both lose to Hungarian. The advantage of the "correct" OT framing is real but small, and it does not move the verdict.
- **H3 falsified.** `enforce_unique=True` *hurts* in every condition pair (e.g., marginal ε=0.1: 0.4747 → 0.4668). Forcing uniqueness on a soft transport plan throws away mass the plan was deliberately distributing.

## What this means for the repo's stated novelty

`decision/000-charter.md` §"Why this is plausibly novel" said the combination "Sinkhorn-OT + Oklab+saliency cost + photomosaic reconstruction as the NBV signal appears to have no prior art." `decision/005-citation-corrections.md` already had to walk back the citations that motivated this novelty claim. Phase 2 now adds a second hit: **on the only benchmark this repo runs, Sinkhorn-OT is not a useful matcher in the photomosaic-NBV signal**. Hungarian outperforms it reliably across the ε grid, the marginal framing, and the uniqueness toggle.

The repo's remaining defensible claims are:

1. **Numerical stability at small ε.** The log-domain implementation in `matching.py` does not NaN/Inf at ε down to 5e-4 on a 50×50 cost matrix at cost scale O(50). This is a contract independent of the active-vision claim, and the tests added in this PR pin it. It is the part of the repo that maps cleanly onto external work (PythonOT/POT issue #723 / PR #724 on partial OT).
2. **Differentiability.** Sinkhorn's plan is differentiable in the cost matrix; Hungarian's is not. We have not yet used this property — the NBV strategy does not back-propagate through the matcher. Phase 2 holds the SSIM result constant under that omission, so this is wished, not measured.
3. **Saliency-as-marginal is the principled framing.** It still loses to Hungarian here, but it loses *less* than the row-scale framing — which is some empirical support for the decision/003 §"Algorithm sketch" framing being more correct than the mosaicraft-era row-scale hack. It also costs us nothing (cost matrix is cheaper to build without the row-scale multiply).

## Decision

Phase 1 was committed to Sinkhorn-OT as the matching layer (`decision/003`). Phase 2 confirms the empirical cost of that commitment on the only benchmark this repo runs: **a measurable, statistically-significant SSIM gap relative to a Hungarian baseline that ships in `scipy`**.

The decision/003 commitment to "no Hungarian fallback" is **not** rescinded. The repo's runtime path remains Sinkhorn-only. But the README's "novelty" framing is downgraded: the user-facing claim is now "log-domain Sinkhorn-OT + Phase-1/2 ablation harness for photomosaic-driven active vision — Hungarian outperforms on the current toy; the repo's value is the open harness, the log-domain stability tests, and the honest negative result, not the Sinkhorn advantage."

## Phase-3 backlog (not yet decided)

These are options, not commitments:

- **Real images.** Every result above is on a hue-ring toy. Whether Hungarian's win generalizes to natural photos is unknown. Running on a small fixed image set (e.g., the mosaicraft test fixtures or a handful of Open Images V7 crops) would test that.
- **Differentiable NBV.** Use Sinkhorn's plan-gradient w.r.t. cost to learn a NBV scoring head, instead of the hand-rolled `MosaicSsimGainStrategy`. Hungarian cannot enter this experiment by definition.
- **Cross-saliency-formulation.** mosaicraft's `compute_saliency_weights` was designed for Hungarian top-K culling. A saliency formulation designed *for* OT marginals (e.g., entropy-regularised attention maps) may close the gap.
- **POT partial-OT log-domain.** Use the test scaffolding here (`test_matching.py` §numerical-stability) as a reference when reviewing PythonOT/POT PR #724.

## How to apply

1. Read `decision/005-citation-corrections.md` first if you haven't — it explains why the prior-art motivation for Sinkhorn was citation-thin.
2. Read this file before quoting any Phase-1 number in a paper or PR description; Phase 2 supersedes the Phase-1 "Hungarian beat us, Phase-2 will fix it" framing with "Phase 2 confirms Hungarian still wins."
3. The runtime path remains Sinkhorn-only per decision/003. Do not introduce a Hungarian fallback even after this finding.

## Signature

User signature not requested — this is empirical reporting, not a new adoption.
