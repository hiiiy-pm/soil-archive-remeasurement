"""Replay the locked one-pilot choice between original and Olsen-weighted joint selection."""
import argparse
from bisect import bisect_left, bisect_right
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from itertools import combinations
import json
from pathlib import Path
from statistics import NormalDist

from alert_summary import ENDPOINTS, interval
from joint_summary import domain_family_weights
from olsen_run import OUT as PRIOR, auxiliary_exact, build_inputs, comparison_root, loader, run_chunk
from olsen_summary import read_cube
from trials import ROOT, np, pd


OUT = ROOT / "direction" / "pilot_olsen_results"
DEVELOPMENT = {"parkgrass": 196031, "hailun": 326041}
NEW_SEED = 487031
FOCAL = "pilot_olsen"
ORIGINAL = "joint_maximin"
WEIGHTED = "gap_joint"
SIMULTANEOUS = NormalDist().inv_cdf(1-.05/(2*78))
POINTWISE = NormalDist().inv_cdf(.975)


def fixed_root(site, out):
    return PRIOR if site in DEVELOPMENT else out / "fixed_designs"


def cell_frame(frame, panels, auxiliary):
    result = {}
    for panel in panels:
        cells = {}
        for (year, treatment), group in frame.iloc[panel.rows].groupby(["year","treatment"]):
            ids = tuple(group.index)
            cells[year,treatment] = (ids, sum(auxiliary[panel.name][i] for i in ids)/len(ids))
        result[panel.name] = cells
    return result


def sign(value):
    return int(value>0)-int(value<0)


def pilot_choice(domains, paid):
    details, aucs = [], []
    for domain, cells in domains.items():
        by_year = {}
        for (year,treatment),(ids,aux_mean) in cells.items():
            values = [paid[domain][i] for i in ids if i in paid[domain]]
            if values:
                by_year.setdefault(year,[]).append((treatment,aux_mean,sum(values)/len(values)))
        bad, good = [], []
        for means in by_year.values():
            for (_,x1,y1),(_,x2,y2) in combinations(means,2):
                discrepant = sign(x1-x2) != sign(y1-y2)
                (bad if discrepant else good).append(abs(x1-x2))
        auc = None
        if bad and good:
            good.sort()
            # A smaller gap in the discordant group earns two units; an equal gap earns one.
            twice = sum(2*len(good)-bisect_left(good,v)-bisect_right(good,v) for v in bad)
            auc = Fraction(twice, 2*len(bad)*len(good))
            aucs.append(auc)
        details.append(dict(domain=domain,observed_comparisons=len(bad)+len(good),
                            discordant=len(bad),concordant=len(good),informative=(auc is not None),
                            auc_exact=str(auc) if auc is not None else ""))
    overall = (sum(aucs)/len(aucs)) if aucs else None
    weighted = overall is not None and overall>Fraction(1,2)
    reason = ("above_chance" if weighted else "uninformative" if overall is None else
              "tied" if overall==Fraction(1,2) else "below_chance")
    return dict(chosen=WEIGHTED if weighted else ORIGINAL, reason=reason,
                auc=float(overall) if aucs else np.nan,auc_exact=str(overall) if aucs else "",
                informative_domains=len(aucs),
                informative_comparisons=sum(row["observed_comparisons"] for row in details if row["informative"]),
                **{key:sum(row[key] for row in details) for key in ("observed_comparisons","discordant","concordant")}),details


def decisions(site, cfg, folder):
    frame, panels, _, _, references = loader(site)(site)
    domains = cell_frame(frame, panels, auxiliary_exact(site,panels))
    choices, audits = [], []
    for rep in range(200):
        pilot = np.random.default_rng(cfg["seed"]+rep).choice(len(frame),cfg["budgets"][0],replace=False)
        paid = {p.name:{i:references[p.name][i] for i in pilot if p.eligible[i]} for p in panels}
        result, by_domain = pilot_choice(domains,paid)
        choices.append(dict(pilot=rep, paid_records=" ".join(map(str,pilot)), **result))
        audits.extend(dict(pilot=rep,**row) for row in by_domain)
    records = pd.DataFrame(choices)
    records.to_csv(folder/"pilot_decisions.csv",index=False)
    pd.DataFrame(audits).to_csv(folder/"pilot_domains.csv",index=False)
    totals = records.groupby(["chosen","reason"],sort=False).agg(
        n=("pilot","size"), informative_comparisons_mean=("informative_comparisons","mean"),
        observed_comparisons_mean=("observed_comparisons","mean"), auc_mean=("auc","mean")).reset_index()
    totals.to_csv(folder/"decision_summary.csv",index=False)
    return records


def run_fixed(site, out, start, stop, workers):
    if site in DEVELOPMENT:
        raise ValueError("Use the archived fixed paths for the two development sites.")
    root = fixed_root(site,out)
    build_inputs(site,root,trial_seed=NEW_SEED)
    chunks = [(i,min(i+4,stop)) for i in range(start,stop,4)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_chunk,site,a,b,root,NEW_SEED) for a,b in chunks]
        for future in as_completed(futures):
            print(*future.result(),flush=True)


def summarize(site, out):
    root, folder = fixed_root(site,out), out/site
    folder.mkdir(parents=True,exist_ok=True)
    cfg = json.loads((root/site/"config.json").read_text())
    assert cfg["seed"]==DEVELOPMENT.get(site,NEW_SEED) and cfg["replicates"]==200
    choices = decisions(site,cfg,folder)
    # No performance data are opened until every pilot decision is fixed.
    saved_cfg, cube = read_cube(site,root,write_overlap=False)
    assert cfg==saved_cfg
    original, weighted = (cfg["designs"].index(s) for s in (ORIGINAL,WEIGHTED))
    choose_weighted = choices.chosen.eq(WEIGHTED).to_numpy()
    selected = np.where(choose_weighted,weighted,original)
    chosen = cube[np.arange(200),selected]
    cube = np.concatenate([cube,chosen[:,None]],axis=1)
    cfg["designs"] = [*cfg["designs"], FOCAL]
    cfg.update(role="development" if site in DEVELOPMENT else "independent")
    (folder/"config.json").write_text(json.dumps(cfg,indent=2)+"\n",encoding="utf-8")
    # A reference to a fixed path is sufficient to recover every paid set.
    parent = comparison_root(site,root)
    paid_paths = []
    for row in choices.itertuples():
        name = f"draw_{row.pilot:03d}.npz"
        file = root/site/"draws"/name if row.chosen==WEIGHTED else parent/"fixed"/site/name
        with np.load(file) as z:
            path = z["paths"][-1 if row.chosen==WEIGHTED else 0]
            np.testing.assert_array_equal(path[:cfg["budgets"][0]], np.fromstring(row.paid_records,sep=" ",dtype=int))
            np.testing.assert_array_equal(np.sort(path),np.arange(cfg["records"]))
            np.testing.assert_array_equal(z["scores"][-1 if row.chosen==WEIGHTED else 0],chosen[row.pilot])
            paid_paths.append(path)
    np.savez_compressed(folder/"chosen_paths.npz",paths=np.array(paid_paths),weighted=choose_weighted)
    performance,paired,subgroups,subgroup_differences,by_pilot = [],[],[],[],[]
    for j,model in enumerate(cfg["models"]):
        for group,metric in ENDPOINTS:
            weights = domain_family_weights(cfg["coordinates"],group)
            score = cube[:,:,:,j,:,cfg["metrics"].index(metric)]@weights
            assert np.isfinite(score).all() and not score[:,:,-1].any()
            nine = score[:,:,:-1].mean(axis=2)
            for label,filter_ in [(ORIGINAL,~choose_weighted),(WEIGHTED,choose_weighted)]:
                delta = (nine[:,weighted]-nine[:,original])[filter_]
                subgroup_differences.append(dict(chosen=label,model=model,group=group,
                    weighting_better=int(np.sum(delta<0)),equal=int(np.sum(delta==0)),
                    original_better=int(np.sum(delta>0)),**interval(delta,POINTWISE)))
                for k in (original,weighted,len(cfg["designs"])-1):
                    subgroups.append(dict(chosen=label,model=model,group=group,design=cfg["designs"][k],
                                          **interval(nine[filter_,k],POINTWISE)))
            for k,rule in enumerate(cfg["designs"]):
                by_pilot.extend(dict(pilot=i,model=model,group=group,design=rule,loss=loss)
                                for i,loss in enumerate(nine[:,k]))
            periods = [("noncensus",nine)]+[(str(b),score[:,:,k]) for k,b in enumerate(cfg["budgets"])]
            for period,draws in periods:
                adjusted = site not in DEVELOPMENT and period=="noncensus" and model=="kriging"
                key = dict(site=site,model=model,period=period,group=group,metric=metric)
                for k,rule in enumerate(cfg["designs"]):
                    performance.append(dict(**key,design=rule,**interval(draws[:,k],POINTWISE)))
                    if rule!=FOCAL:
                        paired.append(dict(**key,design=rule,interval_family=78 if adjusted else 1,
                                           **interval(draws[:,-1]-draws[:,k],SIMULTANEOUS if adjusted else POINTWISE)))
    data, dif = pd.DataFrame(performance),pd.DataFrame(paired)
    data.to_csv(folder/"performance.csv",index=False)
    dif.to_csv(folder/"paired_differences.csv",index=False)
    pd.DataFrame(subgroups).to_csv(folder/"subgroup_losses.csv",index=False)
    pd.DataFrame(subgroup_differences).to_csv(folder/"subgroup_weighted_minus_original.csv",index=False)
    pd.DataFrame(by_pilot).to_csv(folder/"pilot_losses.csv",index=False)
    domain_scores = cube[:,:,:-1].mean(axis=2)
    domains = []
    for j,model in enumerate(cfg["models"]):
        for g,(domain,family) in enumerate(cfg["coordinates"]):
            for m,metric in enumerate(cfg["metrics"]):
                values = domain_scores[:,:,j,g,m]
                for k,design in enumerate(cfg["designs"]):
                    domains.append(dict(site=site,model=model,domain=domain,family=family,metric=metric,
                        design=design,**interval(values[:,k],POINTWISE)))
    pd.DataFrame(domains).to_csv(folder/"domain_results.csv",index=False)
    main = "period=='noncensus' and model=='kriging'"
    benchmarks = data.query(main).query("design!=@FOCAL")
    best = benchmarks[np.isclose(benchmarks["mean"],benchmarks.groupby("group")["mean"].transform("min"),rtol=1e-13,atol=1e-14)]
    best = best.merge(dif.query(main)[["group","design","mean","lower","upper"]],on=["group","design"],
                      suffixes=("_benchmark","_difference"))
    best.to_csv(folder/"strongest.csv",index=False)
    if site not in DEVELOPMENT:
        fixed = dif.query(main).query("design in [@ORIGINAL,@WEIGHTED]").copy()
        fixed["advantage_established"] = fixed.groupby("design").upper.transform(lambda col:
            (col<=0).all() and (col<0).any())
        fixed.to_csv(folder/"fixed_policy_tests.csv",index=False)
    print(site,choices.groupby(["chosen","reason"]).size().to_dict(),flush=True)
    print(data.query(main).query("design in [@ORIGINAL,@WEIGHTED,@FOCAL]").pivot(index="design",columns="group",values="mean").to_string(),flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["fixed","summarize"])
    parser.add_argument("sites",nargs="+")
    parser.add_argument("--output-dir",type=Path,default=OUT)
    parser.add_argument("--workers",type=int,default=1)
    parser.add_argument("--start",type=int,default=0)
    parser.add_argument("--stop",type=int,default=200)
    args = parser.parse_args()
    if not 0<=args.start<args.stop<=200 or args.workers<1:
        parser.error("Enter a nonempty subset of the 200 pilots and a positive worker count.")
    for site in args.sites:
        if args.action=="fixed":
            run_fixed(site,args.output_dir,args.start,args.stop,args.workers)
        else:
            summarize(site,args.output_dir)
