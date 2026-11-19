"""Paired finite-archive comparison for the two fixed joint sampling policies.

Run after both sites have 200 joint draw files. All differences subtract the
comparator from the focal rule, so a negative value favours the focal rule.
The primary family comprises 2 sites, 19 competitors and 3 prespecified losses.
"""
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
OUT = HERE / "joint_results"
FOCAL = "joint_maximin"
ALTERNATE = "alternate"
NEW = [FOCAL, ALTERNATE]
PRIMARY = {("time", "direction"), ("treatment", "direction"), ("primary", "scaled_mse")}
NORMAL = NormalDist().inv_cdf(0.975)
SIMULTANEOUS = NormalDist().inv_cdf(1 - 0.05 / (2 * 114))


def read_draws(site, cfg):
    old_root = HERE / "results" / site
    new_root = OUT / site
    n = cfg["records"]
    nrep = cfg["replicates"]
    n_old = len(cfg["designs"])
    shape = (len(cfg["budgets"]), len(cfg["models"]), len(cfg["coordinates"]), len(cfg["metrics"]))
    assert nrep == 200 and n_old == 18 and cfg["budgets"][-1] == n
    cube = np.empty((nrep, n_old + 2, *shape))
    for rep in range(nrep):
        name = f"draw_{rep:03d}.npz"
        with np.load(old_root / name) as old, np.load(new_root / name) as joint:
            base = old["scores"]
            new = joint["scores"]
            assert base.shape == (n_old, *shape) and new.shape == (3, *shape)
            pilot = old["pilot"]
            np.testing.assert_array_equal(joint["pilot"], pilot)
            paths = joint["paths"]
            assert paths.shape == (3, n)
            for path in paths:
                np.testing.assert_array_equal(path[:len(pilot)], pilot)
                np.testing.assert_array_equal(np.sort(path), np.arange(n))
            assert np.isfinite(new).all() and not np.any(new[:, -1])
            np.testing.assert_allclose(new[:, 0], np.broadcast_to(base[0, 0], new[:, 0].shape),
                                       rtol=1e-12, atol=1e-13)
            cube[rep, :n_old] = base
            cube[rep, n_old] = new[0]
            # Expected loss over the independent fair initial phase for this
            # pilot. Phase means share their pilot and are never extra repeats.
            cube[rep, n_old + 1] = (new[1] + new[2]) * 0.5
    return cube


def domain_family_weights(coordinates, group):
    weights = np.zeros(len(coordinates))
    families = {"treatment", "time"} if group == "primary" else {group}
    for domain in sorted({domain for domain, _ in coordinates}):
        ids = [i for i, (d, f) in enumerate(coordinates) if d == domain and f in families]
        if ids:
            weights[ids] = 1 / len(ids)
    return weights / weights.sum()


def mean_and_mcse(draws):
    return draws.mean(axis=0), draws.std(axis=0, ddof=1) / np.sqrt(len(draws))


def summarize_site(site):
    old_root = HERE / "results" / site
    cfg = json.loads((old_root / "config.json").read_text(encoding="utf-8"))
    designs = cfg["designs"] + NEW
    models, metrics, coords = cfg["models"], cfg["metrics"], cfg["coordinates"]
    assert models == ["kriging", "ridge"] and metrics[-1] == "native_mse"
    cube = read_draws(site, cfg)
    budgets = [(str(b), slice(i, i + 1)) for i, b in enumerate(cfg["budgets"])]
    budgets += [("noncensus", slice(None, -1)), ("after_pilot", slice(1, -1))]
    perf, paired, domains = [], [], []
    for label, ix in budgets:
        # Reduce budgets within each pilot before estimating Monte Carlo error.
        by_pilot = cube[:, :, ix].mean(axis=2)
        coord_mean = by_pilot.mean(axis=0)
        for di, design in enumerate(designs):
            for mi, model in enumerate(models):
                for j, (domain, family) in enumerate(coords):
                    domains.append([site, model, label, domain, family, design, *coord_mean[di, mi, j]])
        for group in ["time", "treatment", "primary", "interaction"]:
            w = domain_family_weights(coords, group)
            # Native MSE is reported per domain, not across P, K or SOC depths.
            val = np.einsum("rdmjk,j->rdmk", by_pilot[..., :-1], w)
            mean, se = mean_and_mcse(val)
            for mi, model in enumerate(models):
                for k, metric in enumerate(metrics[:-1]):
                    for di, design in enumerate(designs):
                        avg, err = mean[di, mi, k], se[di, mi, k]
                        perf.append([site, model, label, group, metric, design, avg, err,
                                     avg - NORMAL * err, avg + NORMAL * err])
            for focal in NEW:
                fi = designs.index(focal)
                dif, se_diff = mean_and_mcse(val[:, fi:fi + 1] - val)
                for mi, model in enumerate(models):
                    for k, metric in enumerate(metrics[:-1]):
                        adjusted = (focal == FOCAL and (group, metric) in PRIMARY and
                                    model == "kriging" and label == "noncensus")
                        for di, comparator in enumerate(designs):
                            if di == fi:
                                continue
                            d, e = dif[di, mi, k], se_diff[di, mi, k]
                            paired.append([site, model, label, group, metric, focal, comparator,
                                           d, e, d - NORMAL * e, d + NORMAL * e,
                                           d - SIMULTANEOUS * e if adjusted else np.nan,
                                           d + SIMULTANEOUS * e if adjusted else np.nan])
    index = ["site", "model", "budget", "group", "metric"]
    performance = pd.DataFrame(perf, columns=index + ["design", "mean", "mcse", "lower", "upper"])
    pairs = pd.DataFrame(paired, columns=index + ["focal", "comparator", "difference", "mcse",
                                                  "lower", "upper", "simultaneous_lower", "simultaneous_upper"])
    domain = pd.DataFrame(domains, columns=["site", "model", "budget", "domain", "family", "design", *metrics])
    assert pairs.simultaneous_upper.notna().sum() == 3 * 19
    historical = pd.read_csv(old_root / "paired_summary.csv", dtype={"budget": str})
    legacy = performance[performance.design.isin(cfg["designs"])]
    keys = index + ["design"]
    a = legacy.set_index(keys)["mean"].sort_index()
    b = historical.set_index(keys)["mean"].sort_index()
    assert a.index.equals(b.index)
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-13)
    return performance, pairs, domain


def decisions(performance, pairs):
    """Report the fixed components and the best historical baseline per endpoint."""
    best_rows, gate_rows = [], []
    for (site, model, budget), results in performance[performance.budget.isin(
            ["noncensus", "after_pilot"])].groupby(["site", "model", "budget"]):
        means = results.set_index(["group", "metric", "design"])["mean"].sort_index()
        subset = pairs[(pairs.site == site) & (pairs.model == model) & (pairs.budget == budget)]
        diffs = subset.set_index(["group", "metric", "focal", "comparator"]).sort_index()
        for group, metric in [("time", "direction"), ("treatment", "direction"),
                              ("primary", "scaled_mse"), ("primary", "direction")]:
            ranking = means.loc[(group, metric)]
            # Define the original reference from the full nine-budget curve for
            # every period. Never choose a different benchmark for a single draw.
            source = performance[(performance.site == site) & (performance.model == model)
                                 & (performance.budget == "noncensus") & (performance.group == group)
                                 & (performance.metric == metric) & ~performance.design.isin(NEW)]
            comparator = source.loc[source["mean"].idxmin(), "design"]
            for focal in NEW:
                diff = diffs.loc[(group, metric, focal, comparator)]
                best_rows.append(dict(site=site, model=model, budget=budget, group=group,
                                      metric=metric, focal=focal, best_original=comparator,
                                      mean=ranking.loc[focal], original_mean=ranking.loc[comparator],
                                      lower_point_estimate=ranking.loc[focal] < ranking.loc[comparator],
                                      **diff[["difference", "mcse", "lower", "upper",
                                              "simultaneous_lower", "simultaneous_upper"]].to_dict()))
        for focal in NEW:
            steps = [("time", "direction", "contrast_mse"), ("treatment", "direction", "sign_error"),
                     ("primary", "scaled_mse", "sign_error")]
            comparisons = [diffs.loc[(*s[:2], focal, s[2])] for s in steps]
            time = means.loc[("time", "direction")]
            span = time.loc["contrast_mse"] - time.loc["sign_error"]
            share = (time.loc["contrast_mse"] - time.loc[focal]) / span if span else np.nan
            direct = diffs.loc[("time", "direction", focal, "sign_error")]
            row = dict(site=site, model=model, budget=budget, focal=focal,
                       favourable_three_point_estimates=all(d.difference < 0 for d in comparisons),
                       simultaneous_three_component_pass=(all(d.simultaneous_upper < 0 for d in comparisons)
                          if all(pd.notna(d.simultaneous_upper) for d in comparisons) else None),
                       time_gain_fraction_retained=share, direct_time_gain=span,
                       focal_time_gain=time.loc["contrast_mse"] - time.loc[focal],
                       time_minus_direct=direct.difference, time_minus_direct_lower=direct.lower,
                       time_minus_direct_upper=direct.upper,
                       time_minus_direct_simultaneous_lower=direct.simultaneous_lower,
                       time_minus_direct_simultaneous_upper=direct.simultaneous_upper)
            for key, d in zip(["time_vs_magnitude", "treatment_vs_direct", "mse_vs_direct"], comparisons):
                row[key] = d.difference
                row[key + "_simultaneous_lower"] = d.simultaneous_lower
                row[key + "_simultaneous_upper"] = d.simultaneous_upper
            gate_rows.append(row)
    return pd.DataFrame(best_rows), pd.DataFrame(gate_rows)


if __name__ == "__main__":
    parts = [summarize_site(site) for site in ["tcse", "ultuna"]]
    performance, pairs, domain = [pd.concat(t, ignore_index=True) for t in zip(*parts)]
    best, gates = decisions(performance, pairs)
    assert pairs.simultaneous_upper.notna().sum() == 114
    for name, df in [("performance", performance), ("paired", pairs), ("domains", domain),
                     ("best_baselines", best), ("gates", gates)]:
        df.to_csv(OUT / (name + ".csv"), index=False)
    main = gates[(gates.focal == FOCAL) & (gates.model == "kriging") & (gates.budget == "noncensus")]
    print(main.to_string(index=False))
