# mosaicraft-active-vision

> **Status:** pre-Phase-1. No runnable code yet — only decision docs.

A research-stage extension of [mosaicraft][mosaicraft] that turns the
photomosaic reconstruction loss into an **active vision** signal:
given a partial scene observation, choose the next camera viewpoint
that makes the resulting photomosaic improve the fastest.

## What you should read first

This repo is intentionally bare. Before any Python code lands, three
decisions must be approved:

| File | What it decides | Status |
|---|---|---|
| [decision/000-charter.md](decision/000-charter.md) | Scope, non-goals, novelty claim | Draft |
| [decision/001-license.md](decision/001-license.md) | Repository license | Draft (Apache-2.0 recommended) |
| [decision/002-evaluation-metric.md](decision/002-evaluation-metric.md) | The single primary DoD metric | Draft (Mosaic-SSIM-gain recommended) |
| [decision/003-matching-algorithm.md](decision/003-matching-algorithm.md) | Sinkhorn-OT instead of Hungarian | Draft |

## Why this exists (one paragraph)

The literature has Sinkhorn-OT (Cuturi 2013, Rabin 2014), it has
next-best-view planning (GenNBV, CVPR 2024), and it has photomosaic
construction (mosaicraft). It does not have a system that **uses the
photomosaic reconstruction loss as the NBV signal**, with the matching
solved by entropic OT whose marginals carry mosaicraft's saliency
weights. That is the gap this repo intends to occupy.

This claim is **unverified**. We do not publish numeric superiority
claims until ablations in `experiments/benchmark_phase1.py` exist and
beat a random-viewpoint baseline (see decision 002).

## What this is NOT

- Not a robotics control library. Simulation / dataset only.
- Not a general active-vision toolkit. Tied to photomosaic so the
  novelty axis is checkable.
- Not a `mosaicraft` fork — it imports from upstream mosaicraft via a
  git submodule (Phase 1) or PyPI (Phase 2+).

## How to read the codebase (eventually)

Once Phase 1 lands:

```
src/mosaicraft_active_vision/
  matching.py     # Sinkhorn-OT (Hungarian is intentionally absent)
  cost.py         # Reuses mosaicraft features.py + saliency.py
  nbv.py          # Next-best-view loop
  metrics.py      # Mosaic-SSIM-gain (primary DoD)

experiments/
  benchmark_phase1.py   # M1 + ablations, < 1 hour on one GPU

decision/         # All decisions are docs first, code second
```

## License

[MIT](LICENSE). Picked to match `mosaicraft` upstream. See
[decision/001-license.md](decision/001-license.md) for the trade-off
record (Apache-2.0 patent grant was given up; the user signed `mit`).

## Where this came from

Created 2026-05-16 as the outcome of a 4-agent debate (architect /
critic / analyst / document-specialist) over how to extend `mosaicraft`
into the most-innovative direction. See `decision/000-charter.md` for
the debate's record.

[mosaicraft]: https://github.com/hinanohart/mosaicraft
