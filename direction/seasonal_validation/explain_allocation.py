"""Post hoc decomposition of the locked joint and magnitude paths.

Reconstruct the two methods at the original budgets, check the stored scores, and
report each contrast's contribution to joint-minus-magnitude differences in
seasonal sign error and combined scaled MSE.
Positive contributions mean greater loss under joint selection. Counts at the
middle budget describe complete coverage of the records defining a contrast.
"""
import argparse
from itertools import combinations
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "spectral"))
from experiment import conditional, discrepancy_kernel
import numpy as np
import pandas as pd


def contrasts(frame, frozen):
    """Construct the signed contrasts from the two archived cell means."""
    cells = frame.groupby(["stratum", "group", "campaign"]).indices
    frozen = frozen.set_index(["stratum", "group", "early", "late"])
    stored_columns = {key: index for index, key in enumerate(frozen.index)}
    rows, matrix, forecasts = [], [], []
    for one, two in combinations(sorted(cells), 2):
        depth, locality1, date1 = one
        depth2, locality2, date2 = two
        if depth != depth2:
            continue
        eligible = False
        prediction_column = None
        if locality1 == locality2:
            family = "time"
            key = (depth, locality1, date1, date2)
            eligible = bool(frozen.loc[key, "eligible"])
            prediction_column = stored_columns[key]
            positive, negative = two, one
        elif date1 == date2:
            family = "locality"
            positive, negative = one, two
        else:
            continue
        indices_plus, indices_minus = cells[positive], cells[negative]
        vector = np.zeros(len(frame))
        vector[indices_plus] = 1.0 / len(indices_plus)
        vector[indices_minus] = -1.0 / len(indices_minus)
        rows.append(dict(stratum=depth, family=family, plus_group=positive[1],
                         plus_campaign=positive[2], minus_group=negative[1],
                         minus_campaign=negative[2],
                         plus_samples="|".join(frame["sample"].iloc[indices_plus]),
                         minus_samples="|".join(frame["sample"].iloc[indices_minus]),
                         sign_eligible=eligible))
        matrix.append(vector)
        forecasts.append(prediction_column)
    detail = pd.DataFrame(rows)
    time_rows = detail.family.eq("time").to_numpy()
    time_columns = np.array([index for index in forecasts if index is not None])
    np.testing.assert_array_equal(np.sort(time_columns), np.arange(len(frozen)))
    return detail, np.stack(matrix), time_rows, time_columns


def run(source, output):
    frame = pd.read_csv(HERE / "frame.csv")
    reference = pd.read_csv(HERE / "references.csv").set_index("sample")
    assert frame["sample"].is_unique and reference.index.is_unique
    y = reference.loc[frame["sample"], "reference"].to_numpy()
    x = frame.auxiliary.to_numpy()
    fixed = pd.read_csv(source / "contrasts.csv")
    detail, matrix, time_rows, time_columns = contrasts(frame, fixed)
    actual = matrix @ y
    np.testing.assert_allclose(actual[time_rows], fixed.measured_change.to_numpy()[time_columns])
    detail["auxiliary_difference"] = matrix @ x
    detail["measured_difference"] = actual

    # The original MSE gives equal weight to depths and supported contrast
    # families, then averages within a family and scales by auxiliary variance.
    family_count = detail.groupby("stratum").family.transform("nunique")
    contrast_count = detail.groupby(["stratum", "family"]).family.transform("size")
    scales = frame.groupby("stratum").auxiliary.var(ddof=0)
    mse_weight = 1 / (detail.stratum.nunique() * family_count * contrast_count * detail.stratum.map(scales))
    detail["scaled_mse_weight"] = mse_weight
    eligible = detail.loc[detail.sign_eligible, "stratum"]
    sign_weight = 1 / (eligible.nunique() * eligible.map(eligible.value_counts()))
    detail["sign_weight"] = sign_weight.reindex(detail.index, fill_value=0.0)

    cov_frame = frame.rename(columns={"group": "treatment", "stratum": "depth", "campaign": "year"})
    covariance = discrepancy_kernel(cov_frame, x) + np.eye(len(frame))
    summary = pd.read_csv(source / "summary.csv").query("budget == 'partial_mean'").set_index(["metric", "policy"])
    with np.load(source / "replays.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["samples"], frame["sample"])
        budgets = archive["budgets"]
        paths = archive["paths"]
        saved_predictions = archive["predictions"]
        saved_mse = archive["scaled_mse"]
        names = archive["policies"].tolist()
    midpoint = int(budgets[len(budgets) // 2 - 1])
    assert budgets[-1] == len(frame) and midpoint == 13
    detail["coverage_budget"] = midpoint
    detail["pilots"] = paths.shape[0]
    support = (matrix != 0)
    for name in ("joint", "magnitude"):
        policy = names.index(name)
        ranked = np.argsort(paths[:, policy], axis=1)
        covered = (ranked < midpoint).astype(int) @ support.T
        detail[name + "_full_coverage_pilots"] = np.sum(covered == support.sum(axis=1), axis=0)
        error_sum = np.zeros(len(detail))
        wrong_sum = np.zeros(len(detail))
        for pilot, path in enumerate(paths[:, policy]):
            for b, n in enumerate(budgets[:-1]):
                paid = np.sort(path[:n])
                estimate = x + conditional(covariance, paid, y[paid] - x[paid])
                estimate[paid] = y[paid]
                predicted = matrix @ estimate
                np.testing.assert_allclose(predicted[time_rows], saved_predictions[pilot, policy, b, time_columns],
                                           rtol=1e-10, atol=1e-10)
                errors = (predicted - actual) ** 2
                np.testing.assert_allclose(mse_weight @ errors, saved_mse[pilot, policy, b],
                                           rtol=1e-10, atol=1e-10)
                error_sum += errors
                wrong_sum += (np.sign(predicted) != np.sign(actual))
        denominator = paths.shape[0] * (len(budgets) - 1)
        detail[name + "_scaled_mse_contribution"] = mse_weight * (error_sum / denominator)
        detail[name + "_wrong_pct"] = 100 * wrong_sum / denominator
        detail[name + "_sign_contribution_pp"] = detail.sign_weight * detail[name + "_wrong_pct"]
        np.testing.assert_allclose(detail[name + "_scaled_mse_contribution"].sum(), summary.loc[("scaled_mse", name), "mean"])
        np.testing.assert_allclose(detail[name + "_sign_contribution_pp"].sum(), summary.loc[("sign_equal_strata", name), "mean"])
        np.testing.assert_allclose(detail.loc[time_rows, name + "_wrong_pct"], fixed[name + "_wrong_pct"].to_numpy()[time_columns])
    for measure in ("scaled_mse_contribution", "sign_contribution_pp"):
        detail["joint_minus_magnitude_" + measure] = detail["joint_" + measure] - detail["magnitude_" + measure]
    output.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.results, args.output or HERE / "diagnostics" / "path_contributions.csv")

