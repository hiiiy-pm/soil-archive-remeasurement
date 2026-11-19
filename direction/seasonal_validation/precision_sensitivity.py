"""Post hoc sensitivity to the reference-material SD floor reported by Kelson et al.

Kelson et al. (https://doi.org/10.1029/2026GC012921, section 2.3) applied
8.3 per meg as a floor to the SD used for each unknown's uncertainty. Here the
same value is applied to the pooled single-run SD in our existing contrast-limit
formula. All selections, predictions and scoring weights are retained.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

HERE = Path(__file__).resolve().parent
FLOOR = 8.3


def rescore(folder, dest):
    lock = json.loads((HERE / "lock.json").read_text("utf-8-sig"))
    frozen = pd.read_csv(HERE / "limits.csv")
    contrasts = pd.read_csv(folder / "contrasts.csv")
    for key in ("stratum", "group", "early", "late", "n_early", "n_late"):
        np.testing.assert_array_equal(frozen[key], contrasts[key])
    sd = max(FLOOR, lock["single_irms_sd"])
    count_factor = np.sqrt(.5 / frozen.n_early + .5 / frozen.n_late)
    critical = t.ppf(.975, lock["df"])
    np.testing.assert_allclose(critical * lock["single_irms_sd"] * count_factor, frozen.limit)
    limits = critical * sd * count_factor
    changed_season = (frozen.late % 10) != (frozen.early % 10)
    actual = contrasts.measured_change.to_numpy()
    eligible = changed_season & (np.abs(actual) > limits)
    with np.load(folder / "replays.npz", allow_pickle=False) as data:
        pred = data["predictions"]
        np.testing.assert_array_equal(data["policies"], ["joint", "magnitude", "direct_sign", "random",
                                                         "alternate_sign_first", "alternate_mse_first"])
        np.testing.assert_allclose(pred[:, :, -1], np.broadcast_to(actual, pred[:, :, -1].shape),
                                   atol=1e-10, rtol=1e-12)
    wrong = 100. * (np.sign(pred[:, :, :-1]) != np.sign(actual))
    scored = np.concatenate((wrong[:, :4], wrong[:, 4:].mean(axis=1, keepdims=True)), axis=1)
    levels = np.sort(frozen.stratum.unique())
    by_level = {}
    for level in levels:
        mask = eligible & (frozen.stratum == level)
        if np.any(mask):
            by_level[level] = (scored[..., mask].mean(axis=(2, 3)), int(mask.sum()),
                               int((changed_season & (frozen.stratum == level)).sum()))
    if not by_level:
        raise ValueError("No cross-season comparisons exceed the sensitivity limits.")
    combined = np.mean([outcomes[0] for outcomes in by_level.values()], axis=0)
    by_level["equal_strata"] = (combined, int(eligible.sum()), int(changed_season.sum()))
    records = []
    for level, (loss, n, available) in by_level.items():
        for k, policy in enumerate(("joint", "magnitude", "direct_sign", "random", "alternation")):
            diff = loss[:, 0] - loss[:, k]
            half = t.ppf(.975, len(diff)-1) * diff.std(ddof=1) / np.sqrt(len(diff))
            records.append(dict(stratum=level, qualifying=n, cross_season_total=available,
                                single_run_sd=sd, policy=policy, wrong_sign_pct=loss[:, k].mean(),
                                joint_minus_policy_pp=diff.mean(), pointwise_mc_low95=diff.mean()-half,
                                pointwise_mc_high95=diff.mean()+half))
    pd.DataFrame(records).to_csv(dest, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rescore(args.results, args.output or args.results / "precision_sensitivity.csv")

