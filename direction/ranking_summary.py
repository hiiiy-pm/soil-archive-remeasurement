"""Paired finite-archive evaluation of family-separated sign and magnitude selection."""
import argparse
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from joint_summary import domain_family_weights, mean_and_mcse


HERE = Path(__file__).resolve().parent
OUT = HERE / "ranking_results"
SITES = ["tcse", "ultuna", "lanna"]
FOCAL = "split_direct"
NEW = [FOCAL, "pooled_direct", "split_bernoulli"]
ORIGINAL_JOINT = "joint_maximin"
PRIMARY = [("time", "direction"), ("treatment", "direction"), ("primary", "scaled_mse")]
SETTINGS = ["site", "model", "budget", "group", "metric"]
REDUCTIONS = {
    "split_under_direct": (FOCAL, "pooled_direct"),
    "split_under_bernoulli": ("split_bernoulli", ORIGINAL_JOINT),
    "direct_under_split": (FOCAL, "split_bernoulli"),
    "direct_under_pooling": ("pooled_direct", ORIGINAL_JOINT),
}
Z95 = NormalDist().inv_cdf(.975)
FAMILY = 3 * 3 * (22 + 2)
Z216 = NormalDist().inv_cdf(1 - .05 / (2 * FAMILY))


def roots(site):
    if site == "lanna":
        root = HERE / "multiperiod_results"
        return root / "benchmarks" / site, root / "fixed" / site, root / site / "performance.csv"
    return HERE / "results" / site, HERE / "joint_results" / site, HERE / "joint_results" / "performance.csv"


def read_site(site, output, include_new):
    base_dir, joint_dir, _ = roots(site)
    cfg = json.loads((base_dir / "config.json").read_text(encoding="utf8"))
    n = cfg["records"]
    assert len(cfg["designs"]) == 18 and cfg["replicates"] == 200
    assert cfg["models"] == ["kriging", "ridge"] and cfg["budgets"][-1] == n
    designs = cfg["designs"] + [ORIGINAL_JOINT, "alternate"] + (NEW if include_new else [])
    axes = [len(cfg[key]) for key in ["budgets", "models", "coordinates", "metrics"]]
    cube = np.empty((200, len(designs), *axes))
    for r in range(200):
        filename = f"draw_{r:03d}.npz"
        pilot = np.random.default_rng(cfg["seed"] + r).choice(n, cfg["budgets"][0], replace=False)
        with np.load(base_dir / filename) as old, np.load(joint_dir / filename) as joint:
            np.testing.assert_array_equal(pilot, old["pilot"])
            np.testing.assert_array_equal(pilot, joint["pilot"])
            masks = np.unpackbits(old["selections"], axis=-1, count=n)
            np.testing.assert_array_equal(masks.sum(axis=-1), np.broadcast_to(cfg["budgets"], masks.shape[:2]))
            cube[r, :18] = old["scores"]
            fixed = joint["scores"]
            cube[r, 18] = fixed[0]
            cube[r, 19] = (fixed[1] + fixed[2]) / 2
        if include_new:
            with np.load(output / site / filename) as new:
                np.testing.assert_array_equal(pilot, new["pilot"])
                assert list(new["policies"].astype(str)) == NEW
                for p in new["paths"]:
                    np.testing.assert_array_equal(p[:len(pilot)], pilot)
                    np.testing.assert_array_equal(np.sort(p), np.arange(n))
                cube[r, 20:] = new["scores"]
    assert np.isfinite(cube).all() and not np.any(cube[:, :, -1])
    pilot_scores = np.broadcast_to(cube[:, :1, 0], cube[:, :, 0].shape)
    np.testing.assert_allclose(cube[:, :, 0], pilot_scores, atol=1e-12, rtol=1e-12)
    return cfg, designs, cube


def interval(draws, critical=None):
    avg, se = mean_and_mcse(draws)
    return dict(difference=avg, mcse=se, lower=avg - Z95*se, upper=avg + Z95*se,
                simultaneous_lower=avg - critical*se if critical else np.nan,
                simultaneous_upper=avg + critical*se if critical else np.nan)


def summarize_site(site, output, include_new=True):
    cfg, designs, cube = read_site(site, output, include_new)
    periods = [(str(n), slice(i, i+1)) for i, n in enumerate(cfg["budgets"])]
    periods.extend([("noncensus", slice(None, -1)), ("after_pilot", slice(1, -1))])
    model_names, coordinates, metrics = (cfg[key] for key in ["models", "coordinates", "metrics"])
    performance, paired, factorial, domains = [], [], [], []
    comparisons = [(FOCAL, c) for c in designs if c != FOCAL] + [
        ("pooled_direct", ORIGINAL_JOINT), ("split_bernoulli", ORIGINAL_JOINT)] if include_new else []
    if include_new:
        assert len(comparisons) == len(set(comparisons)) == 24
    for period, ix in periods:
        # Each draw contributes one mean over budgets to the across-draw variance.
        means = cube[:, :, ix].mean(axis=2)
        raw = means.mean(axis=0)
        for i, design in enumerate(designs):
            for j, model in enumerate(model_names):
                for k, (domain, family) in enumerate(coordinates):
                    domains.append(dict(site=site, model=model, budget=period, design=design, domain=domain,
                                        family=family, **dict(zip(metrics, raw[i,j,k]))))
        for group in ("time", "treatment", "primary", "interaction"):
            if group != "primary" and not any(f == group for _,f in coordinates):
                continue
            weights = domain_family_weights(coordinates, group)
            estimates = np.einsum("rdmjk,j->rdmk", means[..., :-1], weights)
            avg, se = mean_and_mcse(estimates)
            for j, model in enumerate(model_names):
                for k, metric in enumerate(metrics[:-1]):
                    key = dict(site=site, model=model, budget=period, group=group, metric=metric)
                    for i, design in enumerate(designs):
                        perf, err = avg[i,j,k], se[i,j,k]
                        performance.append(dict(**key, design=design, mean=perf, mcse=err,
                                                lower=perf-Z95*err, upper=perf+Z95*err))
                    if (group, metric) not in PRIMARY or not include_new:
                        continue
                    corrected = Z216 if model=="kriging" and period=="noncensus" else None
                    for focal, comparator in comparisons:
                        differences = estimates[:,designs.index(focal),j,k] - estimates[:,designs.index(comparator),j,k]
                        paired.append(dict(**key,focal=focal,comparator=comparator,**interval(differences,corrected)))
                    if period == "noncensus" and model == "kriging":
                        for name, (focal, comparator) in REDUCTIONS.items():
                            differences = estimates[:,designs.index(focal),j,k] - estimates[:,designs.index(comparator),j,k]
                            factorial.append(dict(**key,effect=name,**interval(differences,corrected)))
                        # Pair within a shared draw before estimating the difference in simple effects.
                        interaction = (estimates[:,designs.index(FOCAL),j,k] - estimates[:,designs.index("pooled_direct"),j,k]
                                       - estimates[:,designs.index("split_bernoulli"),j,k] + estimates[:,designs.index(ORIGINAL_JOINT),j,k])
                        factorial.append(dict(**key,effect="interaction",**interval(interaction)))
    perf = pd.DataFrame(performance)
    old = pd.read_csv(roots(site)[2], dtype={"budget":str})
    old = old[old.site==site].set_index(SETTINGS+["design"]).sort_index()
    check = perf[~perf.design.isin(NEW)].set_index(SETTINGS+["design"]).sort_index()
    assert old.index.equals(check.index)
    np.testing.assert_allclose(check[["mean","mcse"]],old[["mean","mcse"]],atol=1e-12,rtol=1e-12)
    domains = pd.DataFrame(domains)
    domains["native_rmse"] = np.sqrt(domains.native_mse)
    return perf, pd.DataFrame(paired), pd.DataFrame(factorial), domains


def decisions(perf, paired):
    best, claims = [], []
    p = perf[perf.model.eq("kriging") & perf.budget.eq("noncensus")]
    q = paired.dropna(subset=["simultaneous_upper"]).set_index(["site","group","metric","focal","comparator"])
    assert q.index.is_unique and len(q) == FAMILY
    for site,d in p.groupby("site"):
        psite = d.set_index(["group","metric","design"])["mean"].sort_index()
        site_q = q.loc[site].sort_index()
        for group,metric in PRIMARY:
            means=psite.loc[(group,metric)]
            for scope,excluded in [("original18", NEW+[ORIGINAL_JOINT,"alternate"]), ("all22", [FOCAL])]:
                competitors=means.drop(excluded)
                lowest=competitors.idxmin()
                diff=site_q.loc[(group,metric,FOCAL,lowest)]
                best.append(dict(site=site,group=group,metric=metric,scope=scope,
                                 comparator=lowest,comparator_mean=means[lowest],focal_mean=means[FOCAL],**diff.to_dict()))
        row=dict(site=site)
        for label,group,metric,other in [
                ("time_vs_magnitude","time","direction","contrast_mse"),
                ("treatment_vs_joint","treatment","direction",ORIGINAL_JOINT),
                ("mse_vs_direct","primary","scaled_mse","sign_error"),
                ("treatment_vs_direct","treatment","direction","sign_error"),
                ("time_vs_joint","time","direction",ORIGINAL_JOINT),
                ("mse_vs_joint","primary","scaled_mse",ORIGINAL_JOINT)]:
            dif=site_q.loc[(group,metric,FOCAL,other)]
            for k in ["difference","simultaneous_lower","simultaneous_upper"]:
                row[label+"_"+k]=dif[k]
        row["three_components_pass"]=all(row[c+"_simultaneous_upper"]<0 for c in [
            "time_vs_magnitude","treatment_vs_joint","mse_vs_direct"])
        row["old_time_and_size_preserved_zero_margin"]=all(row[c+"_simultaneous_upper"]<=0 for c in ["time_vs_joint","mse_vs_joint"])
        claims.append(row)
    return pd.DataFrame(best),pd.DataFrame(claims)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--check-historical",action="store_true",help="Reproduce prior results using only archived policies.")
    args=parser.parse_args()
    parts=[summarize_site(site,args.output_dir,not args.check_historical) for site in SITES]
    if args.check_historical:
        print("Reproduced all archived means and Monte Carlo standard errors at three trials.")
        return
    perf,pairs,factorial,domain=[pd.concat(x,ignore_index=True) for x in zip(*parts)]
    best,claims=decisions(perf,pairs)
    assert len(factorial)==45 and factorial.simultaneous_upper.notna().sum()==36
    for name,table in [("performance",perf),("paired",pairs),("factorial",factorial),("domains",domain),
                       ("best_baselines",best),("gates",claims)]:
        table.to_csv(args.output_dir/(name+".csv"),index=False)
    print(claims.to_string(index=False))


if __name__=="__main__":
    main()
