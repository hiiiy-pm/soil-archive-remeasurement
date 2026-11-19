"""Describe common-pilot loss concentration; never filter or rerun allocations.

Uses saved scores and pilot indices, with the released auxiliary-data design Z.
All 200 repeats and all 12 allocation labels are retained in the consistency check.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

import pilot_risk

ROOT = Path(__file__).resolve().parents[1]


def run():
    summaries, all_repeats = [], []
    for universe in ("poxc", "weoc", "wen"):
        folder = ROOT / "results" / universe
        config = json.loads((folder / "config.json").read_text(encoding="utf8"))
        _, panels, _ = pilot_risk.load(universe)
        assert len(panels) == 1
        panel = panels[0]
        p = panel.F["year_affine"].shape[1]
        metric = config["metrics"].index("historical_scaled_mse")
        mi_ols = config["methods"].index("year_affine")
        mi_ridge = config["methods"].index("pilot_ridge")
        design_random = config["designs"].index("random")
        design_retained = config["designs"].index("retained_contrast")
        budget_i = config["budgets"].index(config["pilot_n"])
        noncensus = [i for i, n in enumerate(config["budgets"]) if n < config["N"]]
        records = []
        max_design_difference = 0.0
        for path in sorted(folder.glob("draw_*.npz")):
            repeat = int(path.stem.split("_")[-1])
            with np.load(path, allow_pickle=False) as data:
                # Three target families, equal weight for this single-domain frame.
                losses = data["scores"][..., metric].mean(axis=-1)
                common = losses[:, budget_i, :]
                difference = float(np.max(np.abs(common - common[0])))
                max_design_difference = max(max_design_difference, difference)
                # Sequential and batch ridge fits differ at floating-point roundoff.
                # Store the actual maximum discrepancy rather than asserting bitwise identity.
                np.testing.assert_allclose(common, np.broadcast_to(common[0], common.shape),
                                           rtol=1e-9, atol=1e-9)
                pilot = data["pilot"].astype(int)
                assert len(pilot) == config["pilot_n"] and len(set(pilot)) == len(pilot)
                # Known design only: no reference values enter rank/condition diagnosis.
                z = panel.F["year_affine"][pilot]
                rank = int(np.linalg.matrix_rank(z))
                singular = np.linalg.svd(z, compute_uv=False)
                condition_raw = float(np.linalg.cond(z))
                condition = condition_raw if rank == p else float("inf")
                metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf8"))
                settings = metadata["settings"][universe]
                records.append(dict(
                    universe=universe, repeat=repeat, pilot_n=len(pilot), parameter_columns=p,
                    pilot_indices_zero_based=";".join(map(str, pilot.tolist())),
                    ols_pilot_historical_scaled_mse=float(common[0, mi_ols]),
                    ridge_pilot_historical_scaled_mse=float(common[0, mi_ridge]),
                    pilot_design_rank=rank, pilot_design_condition_2=condition,
                    raw_float_condition_2=condition_raw,
                    minimum_singular_value=float(singular[-1]),
                    maximum_singular_value=float(singular[0]),
                    rank_fallback=rank < p, pilot_lambda=float(settings["lam"]),
                    random_ols_noncensus_curve_mse=float(losses[design_random, noncensus, mi_ols].mean()),
                    retained_ols_noncensus_curve_mse=float(losses[design_retained, noncensus, mi_ols].mean()),
                ))
        assert len(records) == 200 and {r["repeat"] for r in records} == set(range(200))
        frame = pd.DataFrame(records)
        ols = frame.ols_pilot_historical_scaled_mse.to_numpy()
        ridge = frame.ridge_pilot_historical_scaled_mse.to_numpy()
        assert np.isfinite(ols).all() and np.isfinite(ridge).all()
        assert (ols >= 0).all() and (ridge >= 0).all()
        order = np.argsort(-ols, kind="stable")
        worst = frame.iloc[order[0]]
        random_curve = float(frame.random_ols_noncensus_curve_mse.mean())
        retained_curve = float(frame.retained_ols_noncensus_curve_mse.mean())
        row = dict(
            universe=universe, repeats=len(frame), pilot_n=config["pilot_n"],
            parameter_columns=p, full_rank_repeats=int((frame.pilot_design_rank == p).sum()),
            rank_fallback_repeats=int(frame.rank_fallback.sum()),
            ols_pilot_mean=float(ols.mean()), ols_pilot_median=float(np.median(ols)),
            ols_pilot_q95=float(np.quantile(ols, .95)), ols_pilot_q99=float(np.quantile(ols, .99)),
            ridge_pilot_mean=float(ridge.mean()), ridge_pilot_median=float(np.median(ridge)),
            ridge_pilot_max=float(ridge.max()),
            max_ols_repeat_zero_based=int(worst["repeat"]), max_ols_pilot_mse=float(ols[order[0]]),
            max_ols_share_of_pilot_total=float(ols[order[0]] / ols.sum()),
            top5_ols_share_of_pilot_total=float(ols[order[:5]].sum() / ols.sum()),
            ridge_mse_at_max_ols_repeat=float(worst.ridge_pilot_historical_scaled_mse),
            rank_at_max_ols_repeat=int(worst.pilot_design_rank),
            condition_2_at_max_ols_repeat=float(worst.pilot_design_condition_2),
            pilot_lambda_at_max_ols_repeat=float(worst.pilot_lambda),
            random_ols_curve_mean=random_curve,
            retained_ols_curve_mean=retained_curve,
            pilot_share_of_random_ols_curve_mean=float(ols.mean() / (len(noncensus) * random_curve)),
            pilot_share_of_retained_ols_curve_mean=float(ols.mean() / (len(noncensus) * retained_curve)),
            max_pilot_repeat_share_of_random_ols_curve_mean=float(ols.max() / (len(frame) * len(noncensus) * random_curve)),
            max_pilot_repeat_share_of_retained_ols_curve_mean=float(ols.max() / (len(frame) * len(noncensus) * retained_curve)),
        )
        summaries.append(row)
        all_repeats.extend(records)
    destination = ROOT / "summaries"
    destination.mkdir(exist_ok=True)
    pd.DataFrame(summaries).to_csv(destination / "tail_diagnostics.csv", index=False)
    pd.DataFrame(all_repeats).to_csv(destination / "tail_diagnostics_repeats.csv", index=False)
    print(pd.DataFrame(summaries)[["universe", "ols_pilot_mean", "max_ols_share_of_pilot_total",
                                  "ridge_mse_at_max_ols_repeat", "condition_2_at_max_ols_repeat",
                                  "pilot_share_of_random_ols_curve_mean"]].to_string(index=False))


if __name__ == "__main__":
    run()
