# 000 — Project Charter

**Status:** DRAFT (week 1, pre-decision)
**Created:** 2026-05-16
**Last user instruction (verbatim, R16):**
> 「最適でossスタートしよう最高に革新的なものを作ろう」
> 「それもエージェントで議論して決めるべき」
> 「最適でエージェントでが出してくれた結論を元に」

## Why this repo exists

The user wants the most innovative OSS achievable by extending `mosaicraft`
(photomosaic generator at /home/runza/oss/mosaicraft) into the robotics domain.
Four parallel agents (architect / critic / analyst / document-specialist) were
asked to choose between three paths:

- **A** — Internal refactor of `mosaicraft` (extract a `matching/` submodule)
- **B** — Small PR to a lucidrains repo (e.g. `vit-pytorch`)
- **C** — A new robotics-flavored OSS that reuses mosaicraft components

Outcome of the debate (seal-then-discuss protocol to detect fake convergence):

| Agent | Sealed verdict | Final verdict |
|---|---|---|
| architect | C | A (sequencing A→C, C blocked on 3 Phase-0 items) |
| critic | B | **REJECT C immediate start** (counter: 3-day B spike + 1 user question) |
| analyst | C | C (only path with self-measurable DoD) |
| document-specialist | C | C, but **replace Hungarian with Sinkhorn-OT** |

The user picked the integrated outcome: **path C with Sinkhorn-OT, week-1
decision docs first, no code until DoD is measurable, `gh repo create` deferred
until Phase 1 works locally.**

## Scope (in)

1. A photomosaic-driven *active vision* loop:
   - Given a partial scene observation, choose the **next-best-view (NBV)**
     such that the photomosaic reconstruction improves the fastest.
2. Reuse from `mosaicraft`:
   - `features.py` (191-dim color+texture descriptor)
   - `saliency.py` (per-cell weighting on a target image)
   - `placement.py::compute_cost_matrix` (Oklab + saliency weighting)
3. New work specific to this repo:
   - **Sinkhorn-OT assignment** replacing Hungarian
     (`scipy.optimize.linear_sum_assignment`).
   - NBV scoring that closes the loop between mosaic reconstruction error and
     camera/viewpoint selection.

## Scope (out, explicit non-goals)

- Real robot arm control (this is a simulation / dataset-driven research repo).
- A "general" active vision library — keep the photomosaic angle so the
  novelty is checkable.
- Hungarian assignment. We deliberately **do not** wrap Hungarian as a
  fallback; the matching submodule is Sinkhorn-only.
- Performance superlatives in README. After the `mosaicraft` "5x faster"
  unverified docstring incident, all numeric claims must cite a
  reproducible benchmark file in `experiments/`.

## Why this is *plausibly* novel (not yet proven)

- Rabin 2014 (HAL hal-01002830, "Adaptive color transfer with relaxed
  optimal transport") uses Sinkhorn for color transfer but does **not** use
  saliency-weighted Oklab features.
- GenNBV (CVPR 2024) is the dominant NBV baseline on Isaac Gym but treats
  reconstruction error as a black box, not as a photomosaic placement loss.
- "Beyond Hungarian" (arXiv:2603.08514, 2026-03) accelerates assignment but
  does not couple it to a saliency-aware viewpoint planner.

The combination — Sinkhorn-OT + Oklab+saliency cost + photomosaic
reconstruction as the NBV signal — appears to have no prior art.
This claim is **unverified until 003-matching-algorithm.md cites concrete
search queries with zero hits**, and until a benchmark exists.

## Decisions still pending (block all code)

- 001 — license
- 002 — evaluation metric (the analyst-mandated self-measurable DoD)
- 003 — matching algorithm details (Sinkhorn hyper-params, complexity budget)

No `src/mosaicraft_active_vision/` Python code is written until 001/002/003
are user-approved.

## Public release gates (block `gh repo create`)

1. 001/002/003 approved.
2. A minimal Phase-1 demo runs end-to-end on a toy dataset and produces a
   plot the user has seen.
3. `verify_attribution.py`-style CI check exists for any vendored data.
4. The README contains **zero unverified numeric claims**.

If any gate fails, the repo stays local under
`/home/runza/oss/mosaicraft-active-vision/`. The repo is **never** pushed
"just to claim the namespace."
