# 003 — Matching Algorithm (Sinkhorn-OT, not Hungarian)

**Status:** SIGNED — Sinkhorn-OT adopted on 2026-05-16.
**Last updated:** 2026-05-16
**Origin:** document-specialist agent — flagged that `mosaicraft`'s
Hungarian assignment is becoming obsolete after
*Beyond Hungarian* (arXiv:2603.08514, March 2026).

## The pivot in one sentence

This repo's matching submodule uses **entropic-regularized optimal
transport** (Sinkhorn) instead of the Jonker-Volgenant Hungarian solver
that `mosaicraft` uses today.

## Why Hungarian is the wrong tool here

1. **Hard assignment is wasteful** when we can revisit cells across
   viewpoints. Sinkhorn produces a soft transport plan that we can
   reuse as a *prior* on the next view's assignment.
2. **Saliency weighting is naturally an OT marginal.** In `mosaicraft`'s
   current code (`placement.py::compute_cost_matrix` line 105-107),
   saliency is multiplied row-wise onto the cost matrix —
   a hack to fake the marginal constraint. OT formalizes it.
3. **GPU-friendly.** `linear_sum_assignment` is single-threaded CPU.
   Sinkhorn (via `geomloss` or a hand-written
   log-domain implementation) runs on GPU and scales to viewpoint
   budgets where Hungarian would OOM.
4. **Recent literature.** *Beyond Hungarian* (2026-03) shows Hungarian
   is not Pareto-optimal anymore for assignment problems with marginal
   constraints — exactly our setting.

## Why entropic-regularized OT specifically (not raw OT)

| Approach | Complexity | GPU? | Differentiable? | Stable? |
|---|---|---|---|---|
| Hungarian (LAP) | $O(n^3)$ | No | No | Yes |
| Raw LP-OT | $O(n^3 \log n)$ | Limited | No | Sensitive |
| **Sinkhorn-OT** | $O(n^2 / \epsilon^2)$ per iter | **Yes** | **Yes** | Yes if log-domain |
| Sliced OT | $O(n \log n)$ | Yes | Yes | Approximate only |

Sinkhorn-OT wins on the (GPU × differentiable × stable) combination.
Differentiability matters because the NBV loop will eventually
back-propagate viewpoint selection through the matching.

## Algorithm sketch

Given:

- cost matrix $C \in \mathbb{R}^{n \times m}$ (cells × tile candidates)
- cell marginal $a \in \Delta^n$ (proportional to **saliency** — this is
  the reuse from `mosaicraft/saliency.py`)
- tile marginal $b \in \Delta^m$ (uniform unless we're penalizing
  overused tiles)
- regularization $\epsilon > 0$

Iterate (log-domain):

```
log_u = log(a) - logsumexp(log_K + log_v[None, :], axis=1)
log_v = log(b) - logsumexp(log_K + log_u[:, None], axis=0)
```

where $\log K = -C / \epsilon$. Stop when $\lVert \log u - \log u^{\text{prev}} \rVert_\infty < \delta$.

Recover transport plan $\pi = \mathrm{diag}(u) \cdot K \cdot \mathrm{diag}(v)$.
Pick per-cell tile by $\arg\max_j \pi_{ij}$.

## Cost matrix C: what goes in it

We **reuse** the components from `mosaicraft` rather than reinventing:

| Component | Source | Reuse mode |
|---|---|---|
| 191-dim color+texture feature | `mosaicraft/features.py` | import as-is |
| Oklab perceptual distance | `mosaicraft/placement.py` L98-103 | extract into `mosaicraft_active_vision/cost.py` |
| Saliency | `mosaicraft/saliency.py` | import as-is, feed into marginal $a$ |

This is the R17 ("既存修正優先") guarantee:
**we add a new repo, but we do not fork mosaicraft.** Mosaicraft stays as
a runtime dependency (or git submodule, see below).

## Dependency choice

| Option | Pros | Cons |
|---|---|---|
| `pip install mosaicraft` | Stable versioning | Mosaicraft must publish to PyPI first |
| Git submodule pinned to commit | No PyPI dependency | Slightly clunkier dev setup |
| Vendored copy of `features.py` etc. | Zero setup | License attribution overhead, drift risk |

Recommend **git submodule** for Phase 1 because mosaicraft's PyPI
release is not confirmed. Switch to pip dependency once mosaicraft
publishes.

## Sinkhorn implementation choice

Two realistic options:

1. **`geomloss.SamplesLoss(loss="sinkhorn", ...)`** — clean API, used in
   research, MIT license. Adds `torch` as a dependency.
2. **Hand-written log-domain Sinkhorn in pure numpy** — no torch
   dependency, but slower and we re-implement well-trodden code.

Recommend (1) for Phase 1 since the NBV loop will want gradients anyway
and torch is unavoidable downstream.

## Hyper-parameter budget (locked for Phase 1)

- $\epsilon \in \{0.01, 0.05, 0.1\}$ — swept in ablation.
- Max iterations: 100.
- Convergence $\delta$: $10^{-4}$.
- Backend: log-domain (numerical stability).

## What this does NOT include

- No Hungarian fallback. The repo refuses to ship a code path that
  silently swaps to `scipy.linear_sum_assignment` — that would let
  benchmark numbers attribute themselves to Sinkhorn while actually
  coming from Hungarian.
- No "approximate Sinkhorn" tricks (Greenkhorn, Nyström) until we have
  baselines.

## Decision

- [x] **Adopt Sinkhorn-OT per this doc** ← adopted
- [ ] Use Hungarian + saliency multiplication (mosaicraft-style)
- [ ] Other

**User signature (verbatim, R16):** `003最適でエージェントが最適とゆってるやつで結果出てた`
— 2026-05-16. Refers to the document-specialist agent's final verdict
(C + Hungarian → Sinkhorn replacement, citing arXiv:2603.08514 "Beyond
Hungarian" 2026-03).

All anti-patterns in the "What this does NOT include" section are now
binding: **no Hungarian fallback path, no approximate Sinkhorn tricks
before baselines exist.**
