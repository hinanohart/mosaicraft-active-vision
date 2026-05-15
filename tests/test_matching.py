"""Tests for ``mosaicraft_active_vision.matching``.

The contract under test is the OT-feasibility one: a converged Sinkhorn
plan must respect both marginals up to the dual-variable tolerance.
This is the property that decision/003 buys us in exchange for moving
away from Hungarian (whose feasibility is exact but discrete).

References
----------
Peyré-Cuturi 2019 §4.2 — feasibility of the Sinkhorn plan.
"""

from __future__ import annotations

import numpy as np
import pytest

from mosaicraft_active_vision.matching import (
    SinkhornResult,
    argmax_assignment,
    sinkhorn_ot,
    uniform_marginal,
)


# ---------------------------------------------------------------------------
# Property: marginals are respected at convergence
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n,m,eps", [(8, 8, 0.05), (12, 9, 0.05), (5, 20, 0.1)])
def test_sinkhorn_marginals_respected(n: int, m: int, eps: float) -> None:
    rng = np.random.default_rng(seed=42)
    cost = rng.uniform(0.0, 1.0, size=(n, m))
    result = sinkhorn_ot(cost, epsilon=eps, max_iter=500, tol=1e-9)

    assert result.converged, f"Sinkhorn did not converge (iter={result.n_iter})"
    plan = result.plan

    a = uniform_marginal(n)
    b = uniform_marginal(m)

    np.testing.assert_allclose(plan.sum(axis=1), a, atol=1e-6)
    np.testing.assert_allclose(plan.sum(axis=0), b, atol=1e-6)
    assert result.final_marginal_error < 1e-6


def test_sinkhorn_non_uniform_marginals() -> None:
    """Saliency-style asymmetric source marginal is respected."""
    rng = np.random.default_rng(seed=7)
    n, m = 6, 6
    cost = rng.uniform(0.0, 1.0, size=(n, m))
    a = np.array([0.05, 0.05, 0.1, 0.2, 0.3, 0.3], dtype=np.float64)
    b = uniform_marginal(m)

    result = sinkhorn_ot(cost, a=a, b=b, epsilon=0.05, max_iter=500, tol=1e-9)
    assert result.converged
    np.testing.assert_allclose(result.plan.sum(axis=1), a, atol=1e-6)
    np.testing.assert_allclose(result.plan.sum(axis=0), b, atol=1e-6)


# ---------------------------------------------------------------------------
# Property: zero-cost pairs absorb mass, high-cost pairs do not
# ---------------------------------------------------------------------------
def test_sinkhorn_concentrates_on_low_cost() -> None:
    """If cell i has cost 0 to tile i and >0 elsewhere, plan should
    place most mass on the diagonal."""
    n = 5
    cost = 1.0 - np.eye(n)  # diagonal zero, off-diagonal one
    result = sinkhorn_ot(cost, epsilon=0.01, max_iter=2000, tol=1e-9)
    assert result.converged

    diag_mass = np.diag(result.plan).sum()
    total_mass = result.plan.sum()
    # At eps=0.01, > 95% of mass concentrates on the diagonal.
    assert diag_mass / total_mass > 0.95


# ---------------------------------------------------------------------------
# Edge cases — guard against silent degradation
# ---------------------------------------------------------------------------
def test_sinkhorn_rejects_zero_marginal() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        sinkhorn_ot(
            cost=np.ones((3, 3)),
            a=np.array([0.0, 0.5, 0.5]),
        )


def test_sinkhorn_rejects_nonfinite_cost() -> None:
    cost = np.ones((3, 3))
    cost[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        sinkhorn_ot(cost)


def test_sinkhorn_rejects_negative_epsilon() -> None:
    with pytest.raises(ValueError, match="epsilon"):
        sinkhorn_ot(cost=np.ones((3, 3)), epsilon=-0.1)


def test_sinkhorn_rejects_non_2d_cost() -> None:
    with pytest.raises(ValueError, match="2-D"):
        sinkhorn_ot(cost=np.ones((3, 3, 3)))


# ---------------------------------------------------------------------------
# Tiny case — minimum-size sanity
# ---------------------------------------------------------------------------
def test_sinkhorn_tiny_2x2() -> None:
    cost = np.array([[0.1, 0.9], [0.9, 0.1]], dtype=np.float64)
    result = sinkhorn_ot(cost, epsilon=0.05, max_iter=200, tol=1e-9)
    assert result.converged
    # diagonal mass dominant
    assert result.plan[0, 0] > result.plan[0, 1]
    assert result.plan[1, 1] > result.plan[1, 0]


# ---------------------------------------------------------------------------
# Assignment recovery
# ---------------------------------------------------------------------------
def test_argmax_assignment_basic() -> None:
    plan = np.array(
        [
            [0.1, 0.7, 0.2],
            [0.6, 0.3, 0.1],
            [0.2, 0.2, 0.6],
        ]
    )
    out = argmax_assignment(plan, enforce_unique=False)
    np.testing.assert_array_equal(out, [1, 0, 2])


def test_argmax_assignment_unique_resolves_conflict() -> None:
    """Two cells both want tile 1; unique mode must split them."""
    plan = np.array(
        [
            [0.1, 0.9, 0.0],
            [0.0, 0.8, 0.2],
            [0.5, 0.0, 0.5],
        ]
    )
    out = argmax_assignment(plan, enforce_unique=True, tie_break="first")
    # Each tile appears at most once
    assert len(set(out.tolist())) == 3


def test_uniform_marginal_signature() -> None:
    a = uniform_marginal(7)
    assert a.shape == (7,)
    assert a.dtype == np.float64
    assert float(a.sum()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Return type structure
# ---------------------------------------------------------------------------
def test_sinkhorn_returns_dataclass_with_required_fields() -> None:
    result = sinkhorn_ot(cost=np.ones((4, 4)), epsilon=0.1, max_iter=50)
    assert isinstance(result, SinkhornResult)
    assert result.plan.shape == (4, 4)
    assert result.log_plan.shape == (4, 4)
    assert result.u.shape == (4,)
    assert result.v.shape == (4,)
    assert isinstance(result.n_iter, int)
    assert isinstance(result.converged, bool)
    assert isinstance(result.final_marginal_error, float)
