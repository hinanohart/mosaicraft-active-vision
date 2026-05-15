"""Log-domain Sinkhorn optimal transport for cell-to-tile assignment.

This is the matching layer mandated by ``decision/003-matching-algorithm.md``.
There is **no Hungarian fallback** — that decision is enforced at the
module boundary by simply not implementing one.

The implementation is in two parts:

1. ``sinkhorn_ot`` -- pure numpy log-domain Sinkhorn iteration. Always
   available, no optional dependencies. Used in CI and as the reference.
2. ``sinkhorn_ot_torch`` -- a torch backend behind ``extras=["gpu"]`` for
   the differentiable NBV loop. Imports are lazy so the numpy backend
   keeps working without torch.

Both backends compute a transport plan ``pi`` with marginals
``(a, b)`` minimizing ``<pi, cost> - eps * H(pi)``, where ``H`` is the
entropy of the plan.

References
----------
Cuturi, M. (2013). *Sinkhorn distances: Lightspeed computation of
    optimal transport.* NIPS.
Feydy, J. et al. (2019). *Interpolating between OT and MMD using
    Sinkhorn divergences.* AISTATS.
Peyré, G. & Cuturi, M. (2019). *Computational optimal transport.*
    Foundations and Trends in Machine Learning, 11(5-6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "SinkhornResult",
    "argmax_assignment",
    "sinkhorn_ot",
    "uniform_marginal",
]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SinkhornResult:
    """Output of a Sinkhorn iteration.

    Attributes
    ----------
    plan : np.ndarray, shape ``(n, m)``
        The transport plan ``pi`` in linear (not log) domain.
    log_plan : np.ndarray, shape ``(n, m)``
        ``log(pi)`` — preferred for numerical reasoning.
    u : np.ndarray, shape ``(n,)``
        Log-scaling for the source side: ``pi = exp(u[:, None] + log_K + v[None, :])``.
    v : np.ndarray, shape ``(m,)``
        Log-scaling for the target side.
    n_iter : int
        Number of Sinkhorn iterations actually run.
    converged : bool
        ``True`` iff the dual variables stopped moving by more than
        ``tol`` before ``max_iter`` was hit.
    final_marginal_error : float
        ``max(|pi.sum(0) - b|.max(), |pi.sum(1) - a|.max())`` after the
        last iteration. Useful as a sanity check in tests.
    """

    plan: NDArray
    log_plan: NDArray
    u: NDArray
    v: NDArray
    n_iter: int
    converged: bool
    final_marginal_error: float


# ---------------------------------------------------------------------------
# Marginal helpers
# ---------------------------------------------------------------------------
def uniform_marginal(n: int) -> NDArray:
    """Return the uniform probability vector of length ``n``."""
    if n <= 0:
        raise ValueError(f"n must be positive; got {n}")
    return np.full(n, 1.0 / n, dtype=np.float64)


def _normalize_marginal(x: NDArray, name: str) -> NDArray:
    """Validate and normalize a marginal vector.

    Marginals must be strictly positive (no zero mass; Sinkhorn diverges
    on a zero coordinate) and sum to one.
    """
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if np.any(arr <= 0):
        raise ValueError(f"{name} must be strictly positive (min={arr.min():.3e})")
    total = float(arr.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"{name} has non-finite or non-positive total {total!r}")
    return arr / total


# ---------------------------------------------------------------------------
# Log-domain Sinkhorn (numpy)
# ---------------------------------------------------------------------------
def sinkhorn_ot(
    cost: NDArray,
    a: NDArray | None = None,
    b: NDArray | None = None,
    *,
    epsilon: float = 0.05,
    max_iter: int = 100,
    tol: float = 1.0e-6,
    return_plan: bool = True,
    stabilize_every: int = 0,
) -> SinkhornResult:
    """Solve entropic-regularized OT in the log domain.

    Optimization problem (Cuturi 2013, Peyré-Cuturi 2019 §4):

        min_{pi >= 0}   <pi, cost> - epsilon * H(pi)
        s.t.  pi @ 1 = a,  pi.T @ 1 = b.

    Iteration (log-domain, "soft-min" form):

        log_u = log(a) - logsumexp(-cost / eps + log_v[None, :], axis=1)
        log_v = log(b) - logsumexp(-cost / eps + log_u[:, None], axis=0)

    The log domain avoids the under/overflow that kills the linear-domain
    Sinkhorn at small ``epsilon``.

    Parameters
    ----------
    cost : np.ndarray, shape ``(n, m)``, float
        Cost matrix. Will be cast to float64.
    a : np.ndarray or None, shape ``(n,)``
        Source marginal (e.g. saliency-derived cell weights). ``None``
        means uniform.
    b : np.ndarray or None, shape ``(m,)``
        Target marginal (e.g. tile-pool weights). ``None`` means uniform.
    epsilon : float, default 0.05
        Entropic regularization strength. Smaller eps = sharper plan,
        more iterations needed, more risk of numerical issues even in
        the log domain. Sweep in ``{0.01, 0.05, 0.1}`` per
        ``decision/002`` ablation 2.
    max_iter : int, default 100
        Hard iteration cap. Matches the Phase-1 budget from
        ``decision/003``.
    tol : float, default 1e-6
        Convergence threshold on ``max(|delta_u|, |delta_v|)``.
    return_plan : bool, default True
        Whether to materialize the (n, m) plan. Set ``False`` if you
        only want the dual variables (e.g. for differentiable OT loss).
    stabilize_every : int, default 0
        If positive, every this many iterations subtract the row-max of
        log_K from the dual variables to keep them in float64 range.
        ``0`` disables stabilization (default safe for eps >= 0.01 with
        normalized cost).

    Returns
    -------
    SinkhornResult
    """
    if cost.ndim != 2:
        raise ValueError(f"cost must be 2-D; got {cost.shape}")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive; got {epsilon}")
    if max_iter <= 0:
        raise ValueError(f"max_iter must be positive; got {max_iter}")
    if tol < 0:
        raise ValueError(f"tol must be non-negative; got {tol}")

    n, m = cost.shape
    cost64 = np.asarray(cost, dtype=np.float64)
    if not np.all(np.isfinite(cost64)):
        raise ValueError("cost contains non-finite entries")

    a_norm = _normalize_marginal(a if a is not None else uniform_marginal(n), "a")
    b_norm = _normalize_marginal(b if b is not None else uniform_marginal(m), "b")
    if a_norm.size != n:
        raise ValueError(f"a has size {a_norm.size}, expected {n}")
    if b_norm.size != m:
        raise ValueError(f"b has size {b_norm.size}, expected {m}")

    log_K = -cost64 / epsilon  # (n, m)
    log_a = np.log(a_norm)
    log_b = np.log(b_norm)

    log_u = np.zeros(n, dtype=np.float64)
    log_v = np.zeros(m, dtype=np.float64)

    converged = False
    last_n_iter = max_iter
    for it in range(max_iter):
        # log_u <- log_a - logsumexp(log_K + log_v[None, :], axis=1)
        new_log_u = log_a - _logsumexp(log_K + log_v[None, :], axis=1)
        # log_v <- log_b - logsumexp(log_K + new_log_u[:, None], axis=0)
        new_log_v = log_b - _logsumexp(log_K + new_log_u[:, None], axis=0)

        du = float(np.max(np.abs(new_log_u - log_u)))
        dv = float(np.max(np.abs(new_log_v - log_v)))
        log_u, log_v = new_log_u, new_log_v

        if stabilize_every > 0 and (it + 1) % stabilize_every == 0:
            # Keep dual variables bounded — pulls out a common shift,
            # plan is unchanged (subtracts from log_u, adds to log_v).
            shift = float(log_u.max())
            log_u = log_u - shift
            log_v = log_v + shift

        if max(du, dv) < tol:
            converged = True
            last_n_iter = it + 1
            break

    log_plan = log_u[:, None] + log_K + log_v[None, :]
    plan = np.exp(log_plan) if return_plan else np.empty((0, 0), dtype=np.float64)

    if return_plan:
        marg_err = max(
            float(np.max(np.abs(plan.sum(axis=1) - a_norm))),
            float(np.max(np.abs(plan.sum(axis=0) - b_norm))),
        )
    else:
        marg_err = float("nan")

    return SinkhornResult(
        plan=plan,
        log_plan=log_plan,
        u=log_u,
        v=log_v,
        n_iter=last_n_iter,
        converged=converged,
        final_marginal_error=marg_err,
    )


# ---------------------------------------------------------------------------
# logsumexp (small inline copy so we don't pull in scipy.special at this layer)
# ---------------------------------------------------------------------------
def _logsumexp(x: NDArray, axis: int) -> NDArray:
    """Numerically stable log-sum-exp along an axis."""
    m = np.max(x, axis=axis, keepdims=True)
    # If a whole slice is -inf (zero-mass marginal -> already rejected above),
    # we still guard so we don't generate NaNs.
    m_finite = np.where(np.isfinite(m), m, 0.0)
    out = np.log(np.sum(np.exp(x - m_finite), axis=axis)) + np.squeeze(m_finite, axis=axis)
    return out


# ---------------------------------------------------------------------------
# Assignment recovery
# ---------------------------------------------------------------------------
def argmax_assignment(
    plan: NDArray,
    *,
    enforce_unique: bool = False,
    tie_break: Literal["first", "highest_mass"] = "first",
) -> NDArray:
    """Recover a hard cell-to-tile assignment from a Sinkhorn plan.

    Parameters
    ----------
    plan : np.ndarray, shape ``(n, m)``
        Transport plan from :func:`sinkhorn_ot`.
    enforce_unique : bool, default False
        If ``True``, no tile is assigned to more than one cell. Cells
        compete for tiles in descending plan-mass order; cells that
        lose pick their next-best unblocked tile.
    tie_break : "first" | "highest_mass"
        How to break ties when multiple cells want the same tile under
        ``enforce_unique``. ``"highest_mass"`` (the default for
        ``enforce_unique=True``) keeps the cell with the larger plan
        mass; ``"first"`` keeps the lower cell index.

    Returns
    -------
    np.ndarray, shape ``(n,)``, int64
        ``out[i]`` is the tile index assigned to cell ``i``.
    """
    if plan.ndim != 2:
        raise ValueError(f"plan must be 2-D; got {plan.shape}")
    n, m = plan.shape
    if not enforce_unique:
        return np.argmax(plan, axis=1).astype(np.int64)

    # Unique-assignment recovery: greedy on plan mass.
    flat = plan.flatten()
    if tie_break == "first":
        order = np.argsort(-flat, kind="stable")  # stable so lower index wins
    else:
        order = np.argsort(-flat, kind="mergesort")

    assigned_cell = np.full(n, -1, dtype=np.int64)
    used_tile = np.zeros(m, dtype=bool)
    for k in order:
        cell, tile = int(k // m), int(k % m)
        if assigned_cell[cell] != -1 or used_tile[tile]:
            continue
        assigned_cell[cell] = tile
        used_tile[tile] = True
        if np.all(assigned_cell != -1):
            break

    if np.any(assigned_cell == -1):
        # Fallback: cells with no available tile pick the highest-mass
        # tile regardless of uniqueness. This only happens when m < n.
        leftover = np.where(assigned_cell == -1)[0]
        for cell in leftover:
            assigned_cell[cell] = int(np.argmax(plan[cell]))
    return assigned_cell
