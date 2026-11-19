"""Describe the saved purchase paths without refitting or retuning any selection rule.

Run from the extracted resource root with
python direction/harvard_forest/explain_replays.py
Pass --replays to use a regenerated replay directory, and --output to redirect the CSVs.
"""
import argparse
from pathlib import Path

from replay import HERE, PATH_NAMES, REPORT_NAMES, conditional, prepare
import numpy as np
import pandas as pd
from scipy.stats import t


def average_orders(values):
    """Average the two alternating orders within a pilot after scoring a loss or count."""
    return np.concatenate((values[:, :4], values[:, 4:].mean(axis=1, keepdims=True)), axis=1)


def loss_summary(per_pilot, case, metric):
    records = []
    for p, name in enumerate(REPORT_NAMES):
        delta = per_pilot[:, 0] - per_pilot[:, p]
        half = t.ppf(.975, len(delta) - 1) * delta.std(ddof=1) / np.sqrt(len(delta))
        records.append(dict(frame=case, metric=metric, policy=name, mean=per_pilot[:, p].mean(),
                            joint_minus_policy=delta.mean(), mc_low=delta.mean() - half,
                            mc_high=delta.mean() + half))
    return records


def score_path(path, budgets, channel, target, truth, contrasts, actual):
    cov = channel.kernel + np.eye(len(truth))
    ref = target.C @ truth
    estimates = []
    mse = []
    for b in budgets:
        ix = np.sort(path[:b])
        pred = channel.x + conditional(cov, ix, truth[ix] - channel.x[ix])
        pred[ix] = truth[ix]
        values = contrasts @ pred
        bought = np.zeros(len(truth), dtype=bool)
        bought[ix] = True
        resolved = ~np.any(contrasts[:, ~bought] != 0, axis=1)
        values[resolved] = actual[resolved]
        estimates.append(values)
        mse.append(target.mse_weights @ ((target.C @ pred - ref) ** 2))
    return np.array(estimates), np.array(mse)


def diagnose(case, replays):
    rows, _, _, channel, target, truth, contrasts, inventory, _ = prepare(case)
    source = replays / case
    with np.load(source / "replays.npz", allow_pickle=False) as saved:
        paths, budgets, predicted = (saved[name] for name in ("paths", "budgets", "predictions"))
        np.testing.assert_array_equal(saved["sample_ids"], rows["sample"].to_numpy())
        np.testing.assert_array_equal(saved["policies"], PATH_NAMES)
        np.testing.assert_array_equal(saved["contrast_ids"], np.arange(1, 1 + len(inventory)))
    assert len(budgets) == 10 and budgets[-1] == len(truth)
    rank = np.argsort(paths, axis=-1)
    np.testing.assert_array_equal(np.sort(paths, axis=-1), np.broadcast_to(np.arange(len(truth)), paths.shape))
    assert np.all(np.sort(paths[:, :, :budgets[0]], axis=2) == np.sort(paths[:, :1, :budgets[0]], axis=2))
    correct = inventory.october_minus_july.to_numpy()
    auxiliary = contrasts @ channel.x
    sign_loss = average_orders(100. * (np.sign(predicted) != np.sign(correct)))
    assert np.all(sign_loss[:, :, -1] == 0)
    average_loss = sign_loss[:, :, :-1].mean(axis=(0, 2))
    published = pd.read_csv(source / "contrast_errors.csv")
    for p, name in enumerate(REPORT_NAMES):
        np.testing.assert_allclose(average_loss[p], published[name + "_error_pct"], atol=1e-10)
    half_budget = budgets[4]
    aux_variance = rows.assign(auxiliary=channel.x).groupby(["site", "horizon"]).auxiliary.var(ddof=0)
    detail = []
    for j, item in inventory.iterrows():
        group = contrasts[j] != 0
        count = average_orders((rank[:, :, group] < half_budget).sum(axis=2)).mean(axis=0)
        for p, name in enumerate(REPORT_NAMES):
            detail.append(dict(frame=case, contrast_id=j + 1, site=item.site, horizon=item.horizon,
                               treatment=item.warming_treatment, eligible_specimens=group.sum(),
                               above_limit=bool(item.above_mean), reference_g_kg=correct[j],
                               loi_estimate_g_kg=auxiliary[j], domain_aux_variance=aux_variance.loc[item.site, item.horizon],
                               policy=name, mid_budget=half_budget,
                               measured_at_mid_budget=count[p], sign_error_pct=average_loss[p, j]))

    # Apply the paper's combined treatment-and-time scaled MSE to the original paths.
    # The kriging fit sees only observations bought at each budget.
    mse = np.zeros(paths.shape[:2] + (len(budgets),))
    for pilot, candidate_paths in enumerate(paths):
        for policy, path in enumerate(candidate_paths):
            check, mse[pilot, policy] = score_path(path, budgets, channel, target, truth, contrasts, correct)
            np.testing.assert_allclose(check, predicted[pilot, policy], rtol=1e-10, atol=1e-10)
    assert np.all(mse[:, :, -1] == 0)
    assert np.all(mse[:, :, 0] == mse[:, :1, 0])
    metrics = loss_summary(average_orders(mse[:, :, :-1].mean(axis=2)), case, "combined_scaled_mse")
    for h in ("Organic", "Mineral"):
        eligible = inventory.above_mean.to_numpy() & (inventory.horizon == h).to_numpy()
        if eligible.any():
            partial = sign_loss[:, :, :-1, eligible].mean(axis=(2, 3))
            metrics += loss_summary(partial, case, h.lower() + "_above_limit_sign_error_pct")
        # Count every purchased specimen in a horizon, including those below a scoring limit.
        used = (rank[:, :, (rows.horizon == h).to_numpy()] < half_budget).sum(axis=2)
        metrics += loss_summary(average_orders(used), case, h.lower() + "_specimens_at_half_budget")
    return detail, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=Path, default=HERE / "results")
    parser.add_argument("--output", type=Path, default=HERE / "diagnostics")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    contrast_rows, outcome_rows = [], []
    for case in ("physical", "duplicate_screen"):
        detail, metrics = diagnose(case, args.replays)
        contrast_rows.extend(detail)
        outcome_rows.extend(metrics)
    pd.DataFrame(contrast_rows).to_csv(args.output / "seasonal_contrasts.csv", index=False)
    pd.DataFrame(outcome_rows).to_csv(args.output / "policy_outcomes.csv", index=False)


if __name__ == "__main__":
    main()
