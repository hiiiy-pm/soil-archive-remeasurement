"""Replay the frozen Harvard Forest comparison and score seasonal directions."""
import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
from scipy.stats import t

from calibrate import HERE, SEASONS, calculate, selected_tins
sys.path.insert(0, str(HERE.parent))
from trials import Channel, TargetSet, discrepancy_kernel, exact_targets, original
from acquisition import select_path
from experiment import conditional
from joint_run import joint_path

PATH_NAMES = ("joint", "magnitude", "direct_sign", "random", "alternate_sign_first", "alternate_mse_first")
REPORT_NAMES = (*PATH_NAMES[:4], "alternation")
SEED = 492701
REPEATS = 200


CASES = ("as_released", "physical", "duplicate_screen")


def eligibility(rows, precision, case):
    """Archive quality flags are prepared separately from selection and scoring."""
    records = rows.merge(selected_tins(rows["sample"]), on="sample", validate="one_to_one")
    tins = records[["tin1", "tin2"]].astype(float)
    bad_mass = ~tins.apply(lambda v: v.between(0, 1000)).all(axis=1)
    h = records.horizon.map(precision)
    cutoff = h.map(lambda p: (np.sqrt(2) * p["single_tin_sd"]
                              * t.ppf(1 - .05 / (2 * len(records)), p["pairs"])))
    bad_precision = (tins.tin1 - tins.tin2).abs() > cutoff
    reason = pd.Series("retained", index=records.index)
    if case != "as_released":
        if case == "duplicate_screen":
            reason.loc[bad_precision] = "tin_disagreement"
        reason.loc[bad_mass] = "carbon_exceeds_physical_range"
        for _, grp in records.groupby(["site", "plot", "horizon"]):
            if (reason.loc[grp.index] != "retained").any():
                reason.loc[grp.index[reason.loc[grp.index] == "retained"]] = "other_season_failed"
        for _, grp in records.groupby(["site", "horizon", "warming_treatment"]):
            left = grp.index[reason.loc[grp.index] == "retained"]
            if len(left) < 4:
                reason.loc[left] = "fewer_than_two_matched_plots"
    records["mass_violation"] = bad_mass
    records["tin_disagreement"] = bad_precision
    records["disagreement_limit"] = cutoff
    records["eligibility"] = reason
    records[["tin1", "tin2"]] = tins
    return records


def prepare(case):
    frozen = json.loads((HERE / "freeze.json").read_text(encoding="utf-8"))
    for name in ("protocol.md", "thresholds.csv", "calibration.json", "sample_roles.csv"):
        raw = (HERE / name).read_bytes()
        lf = raw.replace(b"\r\n", b"\n")
        digests = {hashlib.sha256(content).hexdigest() for content in (raw, lf, lf.replace(b"\n", b"\r\n"))}
        if frozen["sha256"][name] not in digests:
            raise ValueError(f"Frozen file changed: {name}")
    all_rows, _, precision, limits = calculate()
    saved = pd.read_csv(HERE / "thresholds.csv")
    pd.testing.assert_frame_equal(limits, saved, check_exact=False, rtol=1e-12)
    roles = pd.read_csv(HERE / "sample_roles.csv", usecols=["sample", "role"])
    pd.testing.assert_frame_equal(all_rows[["sample", "role"]], roles)
    rows = all_rows[all_rows.role == "validation"].sort_values("sample").reset_index(drop=True)
    audit = eligibility(rows, precision, case)
    rows = rows.loc[audit.eligibility == "retained"].reset_index(drop=True)
    frame = pd.DataFrame(dict(year=rows.timepoint.map(SEASONS), treatment=rows.site + "/" + rows.warming_treatment,
                              plot=rows.site + "/" + rows["plot"].astype(str), depth=rows.horizon,
                              record_id=rows["sample"]))
    x = rows.loi_pct.to_numpy() * rows.horizon.map({h: p["loi_slope"] for h, p in precision.items()}).to_numpy()
    panels = []
    domains = []
    domain_count = rows.groupby(["site", "horizon"]).ngroups
    for (site, horizon), group in rows.groupby(["site", "horizon"]):
        mask = (rows.site == site) & (rows.horizon == horizon)
        domains.append((site, horizon))
        panels.append(original.engine.make_panel(site + "_" + horizon, frame, x, mask.to_numpy(), "g/kg", domain_count))
    channel = Channel("C", np.arange(len(frame)), x, discrepancy_kernel(frame, x), list(range(len(panels))))
    targets = TargetSet(panels, channel)
    # Reference values are available only to the paid oracle and the separate scorer.
    tins = selected_tins(rows["sample"]).set_index("sample").loc[rows["sample"]]
    exact = (tins.tin1 + tins.tin2) / 2
    truth = np.array([float(v) for v in exact])
    reference = dict(enumerate(exact))
    all_contrasts = exact_targets(panels, targets, {p.name: reference for p in panels})
    inventory = []
    contrast_rows = []
    lookup = limits.set_index(["site", "horizon", "warming_treatment"])
    for pi, _, family, start, stop, _ in targets.groups:
        if family != "time":
            continue
        site, horizon = domains[pi]
        for i in range(start, stop):
            _, cell_ids, coeff = targets.definitions[i]
            (late, label), (early, label2) = [panels[pi].cells[j] for j in cell_ids]
            assert (late, early) == (10, 7) and label == label2 and list(coeff) == [1, -1]
            treatment = label.split("/")[1]
            lim = lookup.loc[(site, horizon, treatment)]
            n_july, n_oct = [int(panels[pi].n[j]) for j in reversed(cell_ids)]
            assert n_july == n_oct and n_july >= 2
            mean_limit = lim.specimen_limit / np.sqrt(n_july)
            if n_july == lim.plots:
                assert np.isclose(mean_limit, lim.mean_limit)
            measured = all_contrasts[i]
            contrast_rows.append(i)
            inventory.append(dict(site=site, horizon=horizon, warming_treatment=treatment, plots=n_july,
                                  october_minus_july=float(measured), reference_exact=str(measured),
                                  mean_limit=mean_limit, specimen_limit=lim.specimen_limit,
                                  nonzero=bool(measured), above_mean=abs(float(measured)) > mean_limit,
                                  above_specimen=abs(float(measured)) > lim.specimen_limit))
    inventory = pd.DataFrame(inventory)
    if case == "as_released":
        assert len(inventory) == 8 and len(frame) == 42
    assert np.allclose(targets.C[contrast_rows] @ truth, inventory.october_minus_july, rtol=1e-12)
    return rows, frame, panels, channel, targets, truth, targets.C[contrast_rows], inventory, audit


def make_paths(frame, panels, channel, targets, truth, pilot, seed):
    name = panels[0].name
    result = []
    for policy in PATH_NAMES:
        if policy == "random":
            left = np.setdiff1d(np.arange(len(frame)), pilot)
            path = np.r_[pilot, np.random.default_rng(seed + 200000).permutation(left)]
        else:
            oracle = original.PaidOracle({name: truth})
            paid = oracle.reveal(pilot)
            if policy in ("magnitude", "direct_sign"):
                rule = "contrast_mse" if policy == "magnitude" else "sign_error"
                path = select_path(frame, [channel], [targets], 1., pilot, paid, oracle.reveal,
                                   panels, rule, seed + 100000)
            else:
                rule = "maximin" if policy == "joint" else policy
                path = joint_path(len(frame), panels, [channel], [targets], 1., pilot, paid, oracle.reveal,
                                  rule, seed + 100000)
            assert len(oracle.paid) == len(frame)
        np.testing.assert_array_equal(path[:len(pilot)], pilot)
        np.testing.assert_array_equal(np.sort(path), np.arange(len(frame)))
        result.append(path)
    return result


def run(output, case):
    rows, frame, panels, channel, target, truth, contrasts, inventory, audit = prepare(case)
    output.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output / "eligibility.csv", index=False)
    budgets = np.array([(len(frame) * i + 5) // 10 for i in range(1, 11)])
    cov = channel.kernel + np.eye(len(frame))
    actual = inventory.october_minus_july.to_numpy()
    paths = np.empty((REPEATS, len(PATH_NAMES), len(frame)), dtype=np.int16)
    predictions = np.empty((REPEATS, len(PATH_NAMES), len(budgets), len(actual)))
    for j in range(REPEATS):
        seed = SEED + j
        pilot = np.random.default_rng(seed).choice(len(frame), budgets[0], replace=False)
        paths[j] = make_paths(frame, panels, channel, target, truth, pilot, seed)
        for k, path in enumerate(paths[j]):
            for b, n in enumerate(budgets):
                ix = np.sort(path[:n])
                paid = np.zeros(len(frame), dtype=bool)
                paid[ix] = True
                pred = channel.x + conditional(cov, ix, truth[ix] - channel.x[ix])
                pred[ix] = truth[ix]
                values = contrasts @ pred
                resolved = ~np.any(contrasts[:, ~paid] != 0, axis=1)
                values[resolved] = actual[resolved]
                predictions[j, k, b] = values
        assert np.all(predictions[j, :, 0] == predictions[j, 0, 0])
    assert np.isfinite(predictions).all() and np.all(predictions[:, :, -1] == actual)
    np.savez_compressed(output / "replays.npz", paths=paths, predictions=predictions, budgets=budgets,
                        policies=np.array(PATH_NAMES), sample_ids=rows["sample"].to_numpy(dtype=str),
                        pilot_seeds=np.arange(SEED, SEED + REPEATS), contrast_ids=np.arange(1, len(actual) + 1))
    inventory.insert(0, "contrast_id", np.arange(1, len(actual) + 1))
    inventory.to_csv(output / "contrast_inventory.csv", index=False)
    summarize(predictions, budgets, inventory, output)


def summarize(pred, budgets, inventory, output):
    truth = inventory.october_minus_july.to_numpy()
    wrong = (np.sign(pred) != np.sign(truth)) * 100.
    losses = np.concatenate([wrong[:, :4], wrong[:, 4:].mean(axis=1, keepdims=True)], axis=1)
    rows = []
    per_contrast = losses[:, :, :-1, :].mean(axis=(0, 2)).T
    detail = inventory.copy()
    for k, name in enumerate(REPORT_NAMES):
        detail[name + "_error_pct"] = per_contrast[:, k]
    detail.to_csv(output / "contrast_errors.csv", index=False)
    for gate in ("above_mean", "above_specimen", "nonzero"):
        selected = inventory[gate].to_numpy(dtype=bool)
        if not selected.any():
            continue
        per_budget = losses[:, :, :, selected].mean(axis=-1)
        summaries = [(str(b), per_budget[:, :, i]) for i, b in enumerate(budgets)]
        summaries.append(("partial_mean", per_budget[:, :, :-1].mean(axis=2)))
        for budget, values in summaries:
            for k, name in enumerate(REPORT_NAMES):
                diff = values[:, 0] - values[:, k]
                mean = float(diff.mean())
                half = t.ppf(.975, len(diff)-1) * diff.std(ddof=1) / np.sqrt(len(diff))
                rows.append(dict(gate=gate, budget=budget, n_contrasts=int(selected.sum()), comparator=name,
                                 joint_pct=values[:, 0].mean(), comparator_pct=values[:, k].mean(),
                                 joint_minus_comparator_pp=mean, mc_low_pp=mean-half, mc_high_pp=mean+half))
    results = pd.DataFrame(rows)
    results.to_csv(output / "summary.csv", index=False)
    print(results[(results.budget == "partial_mean") & (results.comparator == "magnitude")].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--case", choices=(*CASES, "all"), default="all")
    args = parser.parse_args()
    for case in (CASES if args.case == "all" else (args.case,)):
        print(case, flush=True)
        run(args.output / case, case)
