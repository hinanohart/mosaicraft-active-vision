# 002 — Evaluation Metric (self-measurable DoD)

**Status:** SIGNED — M1 adopted on 2026-05-16.
**Last updated:** 2026-05-16
**Origin:** analyst agent precondition — *"C のみ自前で測定可能な数値 DoD を持つ"*. This file makes that DoD concrete.

## What this file decides

The single primary metric this repo will report in every benchmark, every
PR description, and the README. Secondary metrics may exist for diagnostics
but must never be confused with the headline number.

## What we are measuring

Active vision = given a partial scene observation, choose the next camera
viewpoint such that some reconstruction error goes down fastest. We need a
*reconstruction error* that is photomosaic-flavored, not generic NeRF.

## Candidate metrics

### M1 — "Mosaic SSIM gain per view" (proposed primary)

For a fixed target image $T$ and a tile pool sampled from the partial scene
observed so far:

$$
\Delta\text{SSIM}_k = \text{SSIM}(\text{Mosaic}_k, T) - \text{SSIM}(\text{Mosaic}_{k-1}, T)
$$

where $\text{Mosaic}_k$ is the photomosaic generated after the $k$-th
viewpoint has been visited. Sample efficiency = how many views to reach a
fixed SSIM threshold.

**Why this is the right primary:**

- Directly grounded in the photomosaic application — keeps the repo's
  novelty axis (photomosaic-as-perception-task) front and center.
- Self-measurable: no oracle, no human label, only `SSIM` from
  `skimage.metrics`.
- Comparable to GenNBV: we can also report **standard NBV coverage** as a
  secondary number so reviewers can sanity-check.

**Risks / unknowns:**

- SSIM depends on the choice of target image $T$. We must publish the
  benchmark target set (small, fixed, in repo).
- The metric is undefined for very early $k$ when the tile pool is too
  small to tile the target. We will only score from $k$ such that
  $|\text{tiles}| \ge n_\text{cells}$.

### M2 — "View coverage" (secondary, diagnostic)

Standard NBV metric: fraction of scene voxels observed.
**Why secondary:** measurable but identical to GenNBV's metric — gives
us no novelty signal, only "are we competitive."

### M3 — "Reconstruction MSE on held-out NeRF" (rejected)

**Why rejected:** requires training a NeRF, which inflates compute budget
and pulls us toward the NeRF community whose metric definitions are
contested. Out of scope.

### M4 — "User-rated mosaic quality" (rejected)

**Why rejected:** not self-measurable; can't run in CI.

## Required ablations (block "Phase 1 done" claim)

To trust M1, we must run and publish:

1. **Random viewpoint baseline.** If M1 doesn't beat random, we don't ship.
2. **Sinkhorn vs. Hungarian on identical viewpoints.** Confirms (or
   refutes) the doc-specialist's hypothesis that the matching change
   alone moves M1.
3. **Saliency on/off.** Confirms that mosaicraft's saliency reuse
   actually contributes.
4. **Oklab on/off.** Same justification.

If any of (1)-(4) shows no signal, we revise the charter before publishing.

## Compute budget for Phase 1

- Dataset: 1 toy scene + 4 target images.
- Viewpoint set: 32 candidate poses on a sphere.
- Sinkhorn iterations: ≤ 100, epsilon swept over {0.01, 0.05, 0.1}.
- One full ablation run must complete in **< 1 hour on a single GPU**.
  If it doesn't, the metric or the matching algorithm needs to change
  before Phase 1.

## Decision

- [x] **M1 as primary, M2 reported as diagnostic** ← adopted
- [ ] M2 as primary (we abandon the photomosaic novelty axis)
- [ ] Custom metric

**User signature (verbatim, R16):** `002は最適で` — 2026-05-16.
The phrase "最適で" refers to the recommended option above (M1 primary,
M2 diagnostic), confirmed by the user picking the same form for 003.

All four ablations (random baseline / Sinkhorn vs. Hungarian / saliency on-off /
Oklab on-off) remain required for the "Phase 1 done" claim. No `gh repo
create` until all four pass.

## Where this metric lives in code

- `src/mosaicraft_active_vision/metrics.py` — `mosaic_ssim_gain(...)`.
- `experiments/benchmark_phase1.py` — runs M1 + ablations, writes JSON.
- `tests/test_metrics.py` — golden-hash check on a tiny fixed scene.

None of the above is written until this file is approved.
