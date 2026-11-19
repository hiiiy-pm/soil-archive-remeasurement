"""Choose once from pilot observations by comparing two hypothetical paid assays."""
from copy import copy
from itertools import product

from acquisition import PosteriorState, sign_gain
from joint_run import COMPONENTS, choose
from trials import np


def updated(state, local, value):
    # The branch shares immutable target definitions with its parent.
    branch = copy(state)
    for name in ('R', 'pred', 'mu', 'cross', 'var', 'available', 'unobserved'):
        setattr(branch, name, getattr(state, name).copy())
    branch.update(local, value)
    return branch


def family_weights(targets):
    families = ('time', 'treatment')
    counts = {f: sum(f == group[2] for t in targets for group in t.groups) for f in families}
    if min(counts.values()) == 0:
        raise ValueError('Both treatment and time comparisons are required.')
    weights = []
    for t in targets:
        w = np.zeros((2, len(t.primary)))
        for _, _, family, start, end, _ in t.groups:
            if family in counts:
                w[families.index(family), start:end] = 1 / (counts[family] * (end - start))
        weights.append(w[:, t.primary])
    return weights


def gains_for_selection(states, n):
    out = np.zeros((2, n))
    for pair in states:
        for k, state in enumerate(pair):
            rows, gains = state.gains()
            out[k, rows] += gains
    return out


def endpoint_gains(states, row, lookups, weights, magnitude_gain):
    """One candidate's analytic decreases in sign error and matched MSE."""
    result = np.array([0., 0., magnitude_gain])
    for pair, lookup, w in zip(states, lookups, weights):
        if row not in lookup:
            continue
        state = pair[0]
        local = lookup[row]
        use = state.unobserved > 0
        var = state.var[use]
        z = np.abs(state.mu[use]) / np.sqrt(state.scale * var)
        r = np.abs(state.cross[use, local]) / np.sqrt(var * state.R[local, local])
        result[:2] += w[:, use] @ sign_gain(z, r)
    return result


def pilot_choice(frame, panels, channels, targets, ratio, pilot, initial, seed):
    """Return a fixed course and 2 x 3 gains. Only paid references are accepted."""
    n = len(frame)
    if any(len(v) != len(pilot) for v in initial.values()):
        raise ValueError('Supply only the references from the paid pilot.')
    ids = np.flatnonzero(~np.isin(np.arange(n), pilot))
    if len(ids) < 2:
        raise ValueError('The policy comparison requires two remaining assays.')
    states = [[PosteriorState(ch, t, ratio, pilot, initial[panels[ch.panel_ids[0]].name], rule)
               for rule in COMPONENTS] for ch, t in zip(channels, targets)]
    weights = family_weights(targets)
    lookups = [{int(row): k for k, row in enumerate(c.rows)} for c in channels]
    priority = np.empty(n, dtype=int)
    priority[np.random.default_rng(seed).permutation(n)] = np.arange(n)
    gains = gains_for_selection(states, n)
    first_steps = [choose(ids, gains, priority, c) for c in (None, 0, 1)]
    totals = np.zeros((3, 3))
    nodes = ((-np.sqrt(3.), 1/6), (0., 2/3), (np.sqrt(3.), 1/6))
    # Cache hypothetical first observations shared by several courses.
    for first in sorted(set(first_steps)):
        courses = [k for k, i in enumerate(first_steps) if i == first]
        totals[courses] = endpoint_gains(states, first, lookups, weights, gains[1, first])
        remaining = ids[ids != first]
        measured = [(k, lookup[first]) for k, lookup in enumerate(lookups) if first in lookup]
        for scenario in product(nodes, repeat=len(measured)):
            branch = list(states)
            probability = 1.
            for (k, local), (node, mass) in zip(measured, scenario):
                base = states[k][0]
                hypothetical = base.pred[local] + node * np.sqrt(base.scale * base.R[local, local])
                branch[k] = [updated(state, local, hypothetical) for state in states[k]]
                probability *= mass
            next_gains = gains_for_selection(branch, n)
            for k in courses:
                second = choose(remaining, next_gains, priority, (None, 1, 0)[k])
                totals[k] += probability * endpoint_gains(branch, second, lookups, weights,
                                                          next_gains[1, second])
    expected = np.stack([totals[0], (totals[1] + totals[2]) / 2])
    if not np.isfinite(expected).all() or (expected < 0).any():
        raise ArithmeticError('Invalid pilot-predicted endpoint gains.')
    best = expected.max(axis=0)
    fractions = np.divide(expected, best, out=np.ones_like(expected), where=best > 0)
    joint, alternating = fractions.min(axis=1)
    pick_joint = joint > alternating and not np.isclose(joint, alternating, rtol=1e-12, atol=1e-15)
    return ('joint_maximin' if pick_joint else 'alternate'), expected
