"""Post hoc controls for two kinds of horizon imbalance in the Harvard Forest replay.

Cross two acquisition settings (free choice and a common proportional horizon quota)
with two sign scores (equal qualifying contrasts and equal horizons). The quota uses
only the known horizon inventory, shared pilot and seeded tie breaks. The underlying
assays, exclusions, cutoffs, budgets and kriging model remain those of each saved frame.
The combined scaled MSE retains the seven-archive domain and family weights.
"""
import argparse
from pathlib import Path

from replay import HERE, PATH_NAMES, original, prepare
import numpy as np
import pandas as pd
from acquisition import PosteriorState
from joint_run import choose, tied_best
from explain_replays import average_orders, loss_summary, score_path


def quota_sequence(organic, pilot, seed):
    """Choose the next horizon that makes cumulative counts closest to frame proportions."""
    total = len(organic)
    n_organic = int(organic.sum())
    observed = int(organic[pilot].sum())
    tie_rng = np.random.default_rng(seed + 300000)
    remaining_horizons = []
    for n in range(len(pilot) + 1, total + 1):
        candidates = []
        if (n - 1) - observed < total - n_organic:
            candidates.append(0)
        if observed < n_organic:
            candidates.append(1)
        distances = [abs((observed + h) * total - n * n_organic) for h in candidates]
        tied = [h for h, distance in zip(candidates, distances) if distance == min(distances)]
        selected = tied[0] if len(tied) == 1 else int(tie_rng.choice(tied))
        remaining_horizons.append(selected)
        observed += selected
    return np.array(remaining_horizons, dtype=bool)


def constrained_path(organic, pilot, schedule, channel, target, policy, seed, oracle):
    n = len(organic)
    priority = np.empty(n, dtype=int)
    fixed = np.random.default_rng(seed + 100000).permutation(n)
    priority[fixed] = np.arange(n)
    remaining = np.ones(n, dtype=bool)
    remaining[pilot] = False
    pilot_values = oracle.reveal(pilot)["C"]
    states = []
    if policy == "random":
        random_order = np.random.default_rng(seed + 200000).permutation(np.flatnonzero(remaining))
        priority[random_order] = np.arange(len(random_order))
    else:
        rules = {"direct_sign": ("sign_error",), "magnitude": ("contrast_mse",)}
        for rule in rules.get(policy, ("sign_variance", "contrast_mse")):
            states.append(PosteriorState(channel, target, 1., pilot, pilot_values, rule))
    path = list(pilot)
    for step, horizon in enumerate(schedule):
        allowed = remaining if horizon is None else remaining & (organic == horizon)
        ids = np.flatnonzero(allowed)
        if policy == "random":
            choice = ids[np.argmin(priority[ids])]
        else:
            gains = np.zeros((len(states), n))
            if policy.startswith("alternate"):
                component = (int(policy == "alternate_mse_first") + step) % 2
                active = (component,)
            else:
                component = None
                active = range(len(states))
            for k in active:
                rows, gains_k = states[k].gains()
                gains[k, rows] = gains_k
            if len(states) == 1:
                tied = tied_best(ids, gains[0, ids])
                choice = tied[np.argmin(priority[tied])]
            else:
                choice = choose(ids, gains, priority, component)
        value = oracle.reveal(np.array([choice]))["C"][0]
        for state in states:
            state.update(choice, value)
        path.append(choice)
        remaining[choice] = False
    np.testing.assert_array_equal(np.concatenate(oracle.calls), path)
    return np.asarray(path, dtype=np.int16)


def summarize(case, allocation, forecasts, mse, actual, inventory, budgets):
    wrong = 100. * (np.sign(forecasts) != np.sign(actual))
    included = inventory.above_mean.to_numpy(dtype=bool)
    horizon = inventory.horizon.to_numpy()
    sign = wrong[..., included].mean(axis=-1)
    horizon_sign = np.mean([wrong[..., included & (horizon == h)].mean(axis=-1)
                            for h in ("Organic", "Mineral")], axis=0)
    results = []
    for metric, values in (("per_contrast_sign_error_pct", sign),
                           ("equal_horizon_sign_error_pct", horizon_sign),
                           ("combined_scaled_mse", mse)):
        assert np.all(values[:, :, 0] == values[:, :1, 0]) and np.all(values[:, :, -1] == 0)
        by_pilot = average_orders(values)
        scored = [(str(n), by_pilot[:, :, b]) for b, n in enumerate(budgets)]
        scored.append(("partial_mean", by_pilot[:, :, :-1].mean(axis=-1)))
        for budget, losses in scored:
            for row in loss_summary(losses, case, metric):
                results.append(dict(allocation=allocation, budget=budget, **row))
    return results


def run(case, replays, output):
    rows, _, _, channel, target, truth, contrasts, inventory, _ = prepare(case)
    actual = inventory.october_minus_july.to_numpy()
    organic = rows.horizon.to_numpy() == "Organic"
    with np.load(replays / case / "replays.npz", allow_pickle=False) as saved:
        free = saved["paths"]
        free_predictions = saved["predictions"]
        budgets = saved["budgets"]
        seeds = saved["pilot_seeds"]
        sample_ids = saved["sample_ids"]
        contrast_ids = saved["contrast_ids"]
        np.testing.assert_array_equal(saved["policies"], PATH_NAMES)
    np.testing.assert_array_equal(sample_ids, rows["sample"].to_numpy())
    paths = np.empty_like(free)
    predictions = np.empty_like(free_predictions)
    mse = np.empty(free.shape[:2] + (len(budgets),))
    free_mse = np.empty_like(mse)
    for j, seed in enumerate(seeds):
        pilot = free[j, 0, :budgets[0]]
        np.testing.assert_array_equal(pilot, np.random.default_rng(seed).choice(len(truth), len(pilot), replace=False))
        schedule = quota_sequence(organic, pilot, seed)
        for k, policy in enumerate(PATH_NAMES):
            if j in (0, len(seeds) - 1):
                unconstrained = constrained_path(organic, pilot, [None] * len(schedule), channel,
                                                 target, policy, seed, original.PaidOracle({"C": truth}))
                np.testing.assert_array_equal(unconstrained, free[j, k])
            path = constrained_path(organic, pilot, schedule, channel, target, policy, seed, original.PaidOracle({"C": truth}))
            np.testing.assert_array_equal(np.sort(path), np.arange(len(truth)))
            np.testing.assert_array_equal(path[:len(pilot)], free[j, k, :len(pilot)])
            np.testing.assert_array_equal(organic[path[len(pilot):]], schedule)
            paths[j, k] = path
            predictions[j, k], mse[j, k] = score_path(path, budgets, channel, target, truth, contrasts, actual)
            free_check, free_mse[j, k] = score_path(free[j, k], budgets, channel, target, truth, contrasts, actual)
            np.testing.assert_allclose(free_check, free_predictions[j, k], atol=1e-10, rtol=1e-10)
    np.savez_compressed(output / (case + "_quota.npz"), paths=paths, predictions=predictions,
                        combined_scaled_mse=mse, budgets=budgets, policies=np.array(PATH_NAMES),
                        sample_ids=sample_ids, contrast_ids=contrast_ids, pilot_seeds=seeds)
    return (summarize(case, "free", free_predictions, free_mse, actual, inventory, budgets)
            + summarize(case, "quota", predictions, mse, actual, inventory, budgets))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=Path, default=HERE / "results")
    parser.add_argument("--output", type=Path, default=HERE / "diagnostics")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in ("physical", "duplicate_screen"):
        rows.extend(run(case, args.replays, args.output))
    pd.DataFrame(rows).to_csv(args.output / "horizon_balance.csv", index=False)

