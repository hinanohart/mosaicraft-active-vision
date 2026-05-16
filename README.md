# mosaicraft-active-vision

> **Status:** Phase-1 baseline measured. All four required ablations
> produce signal (decision/004); the result directions are reported
> verbatim — including the ones that contradict the original hypothesis.

A research-stage extension of [mosaicraft][mosaicraft] that turns the
photomosaic reconstruction loss into an **active vision** signal:
given a partial scene observation, choose the next camera viewpoint
that makes the resulting photomosaic improve the fastest.

The matching layer is **log-domain Sinkhorn entropic optimal
transport** (Cuturi 2013, Peyré-Cuturi 2019). No Hungarian fallback
exists in the runtime path — see `decision/003`.

## Phase-1 result (2026-05-16, 4 targets, 32 viewpoints, 16 NBV steps)

| Condition | mean final SSIM | mean Σ gain |
|---|---|---|
| A1 random baseline                 | 0.3927 ± 0.025 | +0.2331 |
| A1 saliency-biased + Sinkhorn      | 0.4177 ± 0.058 | +0.2581 |
| A2 saliency-biased + Hungarian     | **0.4682 ± 0.040** | **+0.3092** |
| A3 saliency OFF + Sinkhorn         | 0.4375 ± 0.055 | +0.2781 |
| A4 Oklab OFF + Sinkhorn            | 0.4177 ± 0.057 | +0.2581 |

Reproduce: `python experiments/benchmark_phase1.py`. JSON output is
written to `experiments/results/phase1_<timestamp>.json`.

## Phase-2 result (2026-05-16, ε sweep + saliency-as-marginal, statistically tested)

`decision/004` had three hypotheses for why Hungarian beat Sinkhorn
in Phase 1: ε was wrong, saliency was applied as a row-scale instead
of as the OT marginal, and the greedy argmax wasted mass. Phase 2
tests all three at N=4 seeds × N=4 targets with paired bootstrap
95 % CIs.

| Condition (best per family) | mean final SSIM | Δ vs Hungarian (95 % CI) |
|---|---|---|
| **hungarian** | **0.5024 ± 0.037** | — |
| saliency-as-marginal, ε=0.1, greedy | 0.4747 ± 0.046 | −0.028 [−0.041, −0.015] |
| saliency-as-marginal, ε=0.05, greedy | 0.4413 ± 0.046 | −0.061 [−0.073, −0.049] |
| row-scale, ε=0.1, greedy | 0.4609 ± 0.048 | −0.041 [−0.053, −0.029] |

**Every Sinkhorn variant we tried lost to Hungarian** with a 95 % CI
strictly below zero. The "real" OT framing (saliency-as-marginal)
beats the row-scale hack by a hair, but both lose. The honest
verdict — and the discussion of what this repo's defensible claims
actually are after this finding — is in
[`decision/006-phase2-findings.md`](decision/006-phase2-findings.md).

Reproduce: `python experiments/benchmark_phase2.py`.

## Read the decision docs first

| File | What it decides | Status |
|---|---|---|
| [decision/000-charter.md](decision/000-charter.md) | Scope, non-goals, novelty claim | Reference |
| [decision/001-license.md](decision/001-license.md) | MIT | **SIGNED** |
| [decision/002-evaluation-metric.md](decision/002-evaluation-metric.md) | Mosaic-SSIM-gain (M1) + view coverage (M2) | **SIGNED** |
| [decision/003-matching-algorithm.md](decision/003-matching-algorithm.md) | Log-domain Sinkhorn-OT | **SIGNED** |
| [decision/004-phase1-findings.md](decision/004-phase1-findings.md) | Phase-1 empirical baseline + Phase 2 backlog | Reported |

## Why this exists (one paragraph)

The literature has Sinkhorn-OT (Cuturi 2013), it has next-best-view
planning (FisherRF ECCV 2024, ActiveSplat RA-L 2025, GenNBV CVPR
2024), and it has photomosaic construction (mosaicraft). The gap
this repo *plausibly* occupies is: **use the photomosaic
reconstruction loss as the NBV signal**, with matching solved by
entropic OT whose marginals carry mosaicraft's saliency weights.
**Unverified** until the prior-art search promised in
`decision/000-charter.md` §72-73 is completed. Phase 1 establishes
the empirical baseline; Phase 2 is the scientific contribution. The
audit of the citations used to motivate the original decision pivot
lives in
[`decision/005-citation-corrections.md`](decision/005-citation-corrections.md).

## What this is NOT

- Not a robotics control library. Simulation / dataset only.
- Not a general active-vision toolkit. Tied to photomosaic so the
  novelty axis is checkable.
- Not a `mosaicraft` fork — it imports from upstream mosaicraft via a
  git submodule pinned to commit `2918137`.

## Layout

```
src/mosaicraft_active_vision/
  matching.py     # Sinkhorn-OT (Hungarian is intentionally absent)
  cost.py         # Reuses mosaicraft features.py + saliency.py
  nbv.py          # Next-best-view loop, simulator-agnostic
  metrics.py      # Mosaic-SSIM-gain (M1) + view coverage (M2)

tests/            # 45 unit + property tests
experiments/
  benchmark_phase1.py   # M1 + 4 ablations, runs in < 5 min on a CPU
  results/              # JSON outputs (gitignored except baseline)
external/mosaicraft/    # submodule, pinned

decision/         # All decisions are docs first, code second
```

## Install (research mode)

```bash
git clone --recurse-submodules <repo-url>
cd mosaicraft-active-vision
pip install -e ".[dev]"
PYTHONPATH=src:external/mosaicraft/src pytest tests/
PYTHONPATH=src:external/mosaicraft/src python experiments/benchmark_phase1.py
```

`mosaicraft` is intentionally **not** pip-installed — it ships
`opencv-python` while this repo pins `opencv-python-headless`, and
both expose `cv2`. The submodule's `src/` is added to `sys.path` at
import time instead.

## License

[MIT](LICENSE). Matches `mosaicraft` upstream. See
[decision/001-license.md](decision/001-license.md) for the trade-off
record.

## Provenance

Created 2026-05-16 from a 4-agent debate (architect / critic /
analyst / document-specialist). The debate did **not** reach
consensus: the architect ultimately recommended path A
(refactor `mosaicraft` in place), the critic explicitly **rejected**
immediate start of this path and proposed a 3-day spike on a smaller
lucidrains PR instead, the analyst maintained C, and the
document-specialist maintained C with a Hungarian→Sinkhorn
modification. The integrated path (this repo) is closest to
analyst + document-specialist; the critic's risk warning was
partially borne out by `decision/004` (Hungarian outperforms
Sinkhorn on the Phase-1 toy). Full record:
[`decision/000-charter.md`](decision/000-charter.md) and
[`decision/005-citation-corrections.md`](decision/005-citation-corrections.md).

[mosaicraft]: https://github.com/hinanohart/mosaicraft
