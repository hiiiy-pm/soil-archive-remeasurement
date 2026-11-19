"""Determine the Harvard Forest assay limits without inspecting validation C."""
from __future__ import annotations

import csv
import hashlib
import itertools
from fractions import Fraction
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

HERE = Path(__file__).resolve().parent
CARBON = HERE / "source" / "hf449-01-emf-c-n.csv"
AUXILIARY = HERE / "source" / "hf431-01-temp-acclimation.csv"
HASHES = {
    CARBON.name: "9629b0701cfeda437817838de53497f59a86cad31096538556dd2b7595baedb8",
    AUXILIARY.name: "f8a6420706bd1fbfcbea5e25479a7e832a6d8184c5b492d4765bbba9707023ef",
}
SEASONS = {"July": 7, "October": 10}
METADATA = ["sample", "site", "plot", "warming_treatment", "horizon", "timepoint"]
def tin_concentration(row, tin):
    """Micrograms of C divided by milligrams of soil gives grams of C per kilogram."""
    value = Fraction(row[f"ug_C_rep_{tin}"]) / Fraction(row[f"mg_soil_rep_{tin}"])
    archived_fraction = Fraction(row[f"mg_C_per_g_soil_rep_{tin}"])
    if not np.isclose(float(value), 1000 * float(archived_fraction), rtol=5e-6, atol=1e-6):
        raise ValueError(f"Carbon mass and concentration disagree for {row['sample']}, tin {tin}")
    return value


def metadata():
    for path in (CARBON, AUXILIARY):
        if hashlib.sha256(path.read_bytes()).hexdigest() != HASHES[path.name]:
            raise ValueError(f"Source checksum mismatch for {path.name}")
    samples = pd.read_csv(CARBON, usecols=METADATA)
    auxiliary = pd.read_csv(AUXILIARY, usecols=["Sample", "Site", "Plot", "Treatment",
                                                "Horizon", "Timepoint", "LOI_pct"])
    auxiliary.columns = [c.lower() for c in auxiliary]
    data = samples.merge(auxiliary, on="sample", validate="one_to_one", suffixes=("", "_aux"))
    assert len(data) == len(samples) == 69
    for name in ["site", "plot", "horizon", "timepoint"]:
        assert data[name].equals(data[name + "_aux"])
    assert (data.warming_treatment.str.lower() == data.treatment.str.lower()).all()
    assert np.isfinite(data.loi_pct).all() and (data.loi_pct > 0).all()
    assert set(data.timepoint) == set(SEASONS) and set(data.horizon) == {"Organic", "Mineral"}
    assert not data.duplicated(["site", "plot", "horizon", "timepoint"]).any()
    assert data.groupby(["site", "plot"]).warming_treatment.nunique().max() == 1
    return data[METADATA + ["loi_pct"]].copy()


def assign_roles(frame):
    roles = pd.Series("unmatched", index=frame.index)
    for _, group in frame.groupby(["site", "warming_treatment"]):
        plots = sorted(group["plot"].unique())
        assert len(plots) == 5
        both_months = group.groupby(["plot", "horizon"]).size().unstack(fill_value=0).eq(2)
        def rank(reserved):
            counts = both_months.drop(index=list(reserved)).sum()
            return -int(counts.min()), -int(counts.sum()), reserved

        reserved = min(itertools.combinations(plots, 2), key=rank)
        roles.loc[group.index[group["plot"].isin(reserved)]] = "calibration"
        for (plot, _), rows in group.groupby(["plot", "horizon"]):
            if plot not in reserved and len(rows) == 2:
                roles.loc[rows.index] = "validation"
    return roles


def selected_tins(identifiers):
    selected = set(identifiers)
    with CARBON.open(encoding="utf-8-sig", newline="") as stream:
        rows = [[row["sample"], tin_concentration(row, 1), tin_concentration(row, 2)]
                for row in csv.DictReader(stream) if row["sample"] in selected]
    result = pd.DataFrame(rows, columns=["sample", "tin1", "tin2"])
    assert set(result["sample"]) == selected and result["sample"].is_unique
    return result


def calculate():
    frame = metadata()
    frame["role"] = assign_roles(frame)
    calibration = frame.loc[frame.role == "calibration"].copy()
    held_out = frame.loc[frame.role == "validation"].copy()
    assert set(zip(calibration.site, calibration["plot"])).isdisjoint(zip(held_out.site, held_out["plot"]))
    calibration = calibration.merge(selected_tins(calibration["sample"]), on="sample", validate="one_to_one")
    calibration[["tin1", "tin2"]] = calibration[["tin1", "tin2"]].astype(float)
    assert np.isfinite(calibration[["tin1", "tin2"]]).all().all()
    calibration["difference"] = calibration.tin1 - calibration.tin2
    calibration["duplicate_mean"] = (calibration.tin1 + calibration.tin2) / 2
    precision = {}
    for horizon, group in calibration.groupby("horizon"):
        differences = group.difference.to_numpy()
        m = len(group)
        squares = differences ** 2
        s = float(np.sqrt(squares.sum() / (2 * m)))
        multiplier = float(t.ppf(0.975, m))
        loo = np.sqrt((squares.sum() - squares) / (2 * (m - 1)))
        beta = float((group.loi_pct @ group.duplicate_mean) / (group.loi_pct @ group.loi_pct))
        precision[horizon] = dict(pairs=m, single_tin_sd=s, t_multiplier=multiplier,
                                  specimen_limit=multiplier * s, mean_tin1_minus_tin2=float(differences.mean()),
                                  loo_sd_min=float(loo.min()), loo_sd_max=float(loo.max()), loi_slope=beta)
    limits = []
    for (site, horizon, treatment), group in held_out.groupby(["site", "horizon", "warming_treatment"]):
        july, october = (set(group.loc[group.timepoint == month, "plot"]) for month in SEASONS)
        assert july == october and len(july) >= 2
        n = len(july)
        single = precision[horizon]["specimen_limit"]
        limits.append(dict(site=site, horizon=horizon, warming_treatment=treatment, plots=n,
                           mean_limit=single / np.sqrt(n), specimen_limit=single))
    assert len(limits) == 8
    return frame, calibration, precision, pd.DataFrame(limits)


def export():
    frame, calibration, precision, limits = calculate()
    frame.to_csv(HERE / "sample_roles.csv", index=False, lineterminator="\n")
    calibration.to_csv(HERE / "calibration_pairs.csv", index=False, lineterminator="\n")
    limits.to_csv(HERE / "thresholds.csv", index=False, lineterminator="\n")
    (HERE / "calibration.json").write_text(json.dumps(precision, indent=2) + "\n", encoding="utf-8")
    print(frame.role.value_counts().to_string())
    print(json.dumps(precision, indent=2))
    print(limits.to_string(index=False))


if __name__ == "__main__":
    export()
