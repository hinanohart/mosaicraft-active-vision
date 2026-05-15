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

**Honest reading of these numbers** — see
[`decision/004-phase1-findings.md`](decision/004-phase1-findings.md)
for the full discussion. Headlines:

1. Active-vision premise holds: saliency-biased view selection beats
   random by +0.025 SSIM.
2. On this toy scene with uniform marginals and `epsilon=0.05`,
   Hungarian outperforms Sinkhorn-OT. The OT formulation is **not
   abandoned** (it is the differentiable basis for Phase 2), but
   the README does not claim Sinkhorn wins until the ε sweep and the
   saliency-as-marginal experiment are run.
3. mosaicraft's row-scaling saliency hurts in the OT context — its
   correct usage is as the OT source marginal, deferred to Phase 2.
4. Oklab and the 191-dim LAB features double-count colour; Phase 2
   keeps one or the other, not both.

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
planning (GenNBV, CVPR 2024), and it has photomosaic construction
(mosaicraft). It does not have a system that **uses the photomosaic
reconstruction loss as the NBV signal**, with the matching solved by
entropic OT whose marginals carry mosaicraft's saliency weights.
That is the gap this repo intends to occupy. Phase 1 establishes the
empirical baseline; Phase 2 is the scientific contribution.

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

Created 2026-05-16 as the outcome of a 4-agent debate (architect /
critic / analyst / document-specialist) over how to extend `mosaicraft`
into the most-innovative direction. See `decision/000-charter.md` for
the debate record.

[mosaicraft]: https://github.com/hinanohart/mosaicraft
