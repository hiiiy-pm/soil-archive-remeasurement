"""Bounded integer allocation for the published sum(A_h / n_h) objective.

This project implements the mathematical objective independently. No original
author code is distributed. Scope and adaptations are stated in ALLOCATION.md.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Allocation:
    total: np.ndarray
    new: np.ndarray
    lower: np.ndarray
    fallback: bool
    reason: str
    objective: float
    singleton_census_adaptation: bool


def _integer_vector(x, name):
    z = np.asarray(x)
    if z.ndim != 1 or not np.all(np.isfinite(z)) or np.any(z != np.floor(z)):
        raise ValueError(f'{name} must be a finite integer vector')
    return z.astype(np.int64)


def _metadata_balanced(N, already, budget):
    """Equal-count filling with canonical stratum-order ties; uses no A or y."""
    n = already.copy()
    for _ in range(budget - int(n.sum())):
        eligible = np.flatnonzero(n < N)
        j = eligible[np.argmin(n[eligible])]
        n[j] += 1
    return n


def allocate_a_optimal(N_h, A_h, budget, already=None, *, min_per_stratum=2):
    """Return cumulative and additional allocation with an explicit fallback.

    Parameters
    ----------
    N_h : positive integer vector
        Original stratum sizes for an accumulated-sample objective; use sizes
        of the remaining population for a conditional second-stage objective.
        These are different estimands; this routine does not choose between them.
    A_h : nonnegative finite vector, or NaN for unavailable variance proxies
        N_h**2 times a priority-weighted sum of within-stratum variances.
        Missing variances trigger metadata-only allocation, never become zero.
    budget : integer
        Cumulative allocation for the population supplied as N_h.
    already : nonnegative integer vector
        Actual previously selected counts, default zero.
    min_per_stratum : int
        Default 2 (Wright II). A singleton is taken in full and marked as an
        adaptation. Every budget is returned, including budgets below this floor.

    Notes
    -----
    Equality budget==sum(lower) is feasible mathematically. This deliberately
    avoids the optimall 1.4.0 public WrightII wrapper's <=2H rejection; its
    underlying bounded allocator accepts equality. Exact ties, including all
    zero scores, use canonical stratum index order as R's which.max does.
    """
    N = _integer_vector(N_h, 'N_h')
    a = np.asarray(A_h, dtype=float)
    prev = np.zeros_like(N) if already is None else _integer_vector(already, 'already')
    if a.shape != N.shape or prev.shape != N.shape or len(N) == 0:
        raise ValueError('N_h, A_h and already must have equal nonempty shapes')
    if np.any(N <= 0) or np.any(prev < 0) or np.any(prev > N):
        raise ValueError('Require N_h>0 and 0<=already<=N_h')
    if not np.isscalar(budget) or not np.isfinite(budget) or int(budget) != budget:
        raise ValueError('budget must be an integer')
    budget = int(budget)
    if budget < prev.sum() or budget > N.sum():
        raise ValueError('Budget cannot retract selections or exceed population')
    if int(min_per_stratum) != min_per_stratum or min_per_stratum < 1:
        raise ValueError('min_per_stratum must be a positive integer')
    if np.any(a[np.isfinite(a)] < 0):
        raise ValueError('Variance coefficients cannot be negative')
    floor = np.minimum(N, int(min_per_stratum))
    lower = np.maximum(prev, floor)
    reasons = []
    if budget < lower.sum():
        reasons.append('budget_below_required_floor')
    if not np.all(np.isfinite(a)):
        reasons.append('variance_proxy_unavailable')
    if reasons:
        n = _metadata_balanced(N, prev, budget)
    else:
        n = lower.copy()
        for _ in range(budget - int(n.sum())):
            priority = a / (n.astype(float) * (n + 1))
            priority[n >= N] = -np.inf
            j = int(np.argmax(priority))
            if not np.isfinite(priority[j]):
                raise RuntimeError('No capacity remains before reaching budget')
            n[j] += 1
    if not np.all(np.isfinite(a)):
        objective = float('nan')
    elif np.any((n == 0) & (a > 0)):
        objective = float('inf')
    else:
        objective = float(np.sum(np.divide(a, n, out=np.zeros_like(a), where=n>0)))
    assert int(n.sum()) == budget and np.all(n >= prev) and np.all(n <= N)
    return Allocation(n, n-prev, lower, bool(reasons), ';'.join(reasons) or 'a_optimal',
                      objective, bool(np.any(N < min_per_stratum)))
