"""Compare both Olsen-gap allocations with the shared fixed-budget competitors."""
import argparse
import json
from pathlib import Path
from statistics import NormalDist

from trials import np, pd
from alert_summary import ENDPOINTS, interval
from joint_summary import domain_family_weights
from olsen_run import OUT, comparison_root, loader


FOCAL = "gap_joint"
SIMULTANEOUS = NormalDist().inv_cdf(1-.05/(2*150))
POINTWISE = NormalDist().inv_cdf(.975)


def read_cube(site, out, *, write_overlap=True):
    folder = out/site
    cfg = json.loads((folder/"config.json").read_text())
    named = cfg["designs"]
    root = comparison_root(site,out)
    n = cfg["records"]
    _,panels,_,_,_ = loader(site)(site)
    all_draws, changes, domain_coverage = [], [], []
    for i in range(200):
        file = f"draw_{i:03d}.npz"
        pilot = np.random.default_rng(cfg["seed"]+i).choice(n,cfg["budgets"][0],replace=False)
        with (np.load(root/"benchmarks"/site/file) as base, np.load(root/"fixed"/site/file) as fixed,
              np.load(folder/"draws"/file) as new):
            for obj in (base,fixed,new):
                np.testing.assert_array_equal(obj["pilot"],pilot)
            masks = np.unpackbits(base["selections"],axis=-1,count=n)
            np.testing.assert_array_equal(masks.sum(-1),np.broadcast_to(cfg["budgets"],masks.shape[:2]))
            for path in np.concatenate((fixed["paths"],new["paths"])):
                np.testing.assert_array_equal(path[:len(pilot)],pilot)
                np.testing.assert_array_equal(np.sort(path),np.arange(n))
            score = fixed["scores"]
            blocks = [base["scores"],score[:1],score[1:].mean(axis=0,keepdims=True)]
            if site=="parkgrass":
                with np.load(root/"warning"/file) as extra:
                    np.testing.assert_array_equal(extra["pilot"],pilot)
                    assert list(extra["policies"])==named[20:24]
                    blocks.append(extra["scores"])
            assert list(new["policies"]) == named[-len(new["policies"]):]
            draw = np.concatenate([*blocks,new["scores"]])
            assert len(draw)==26 and np.isfinite(draw).all() and not draw[:,-1].any()
            np.testing.assert_allclose(draw[:,0],np.broadcast_to(draw[0,0],draw[:,0].shape),rtol=0,atol=0)
            all_draws.append(draw)
            old_path, new_path = fixed["paths"][0],new["paths"][-1]
            for design,path in (("joint_maximin",old_path),(FOCAL,new_path)):
                position=np.empty(n,int)
                position[path]=np.arange(1,n+1)
                for panel in panels:
                    domain_coverage.append(dict(pilot=i,design=design,domain=panel.name,
                                                domain_size=len(panel.rows),last_assayed=position[panel.rows].max()))
            for b in cfg["budgets"]:
                overlap = len(set(old_path[:b])&set(new_path[:b]))
                changes.append(dict(pilot=i,budget=b,paid_set_overlap=overlap,
                                    changed_first=int(new_path[len(pilot)]!=old_path[len(pilot)])))
    if write_overlap:
        pd.DataFrame(changes).to_csv(folder/"paid_set_overlap.csv",index=False)
        pd.DataFrame(domain_coverage).to_csv(folder/"domain_completion.csv",index=False)
    return cfg,np.array(all_draws)


def summarize(site,out):
    cfg,cube=read_cube(site,out)
    names=cfg["designs"]
    perf,paired,raw_primary=[],[],[]
    for j,model in enumerate(cfg["models"]):
        for group,metric in ENDPOINTS:
            weights=domain_family_weights(cfg["coordinates"],group)
            values=cube[:,:,:,j,:,cfg["metrics"].index(metric)]@weights
            periods=[("noncensus",values[:,:,:-1].mean(axis=2))]
            periods.extend((str(b),values[:,:,k]) for k,b in enumerate(cfg["budgets"]))
            for period,draws in periods:
                key=dict(site=site,model=model,period=period,group=group,metric=metric)
                primary=(period=="noncensus" and model=="kriging")
                for k,rule in enumerate(names):
                    perf.append(dict(**key,design=rule,**interval(draws[:,k],POINTWISE)))
                    if rule!=FOCAL:
                        paired.append(dict(**key,design=rule,**interval(draws[:,-1]-draws[:,k],
                                              SIMULTANEOUS if primary else POINTWISE)))
                    if primary:
                        raw_primary.extend(dict(pilot=i,group=group,design=rule,loss=v)
                                           for i,v in enumerate(draws[:,k]))
    folder=out/site
    p=pd.DataFrame(perf)
    q=pd.DataFrame(paired)
    p.to_csv(folder/"performance.csv",index=False)
    q.to_csv(folder/"paired_differences.csv",index=False)
    pd.DataFrame(raw_primary).to_csv(folder/"primary_pilot_losses.csv",index=False)
    # Raw native-unit and small-contrast scores stay separate by domain.
    # These intervals describe the observed domains; only the primary family
    # above is used for decisions regarding the two archives.
    noncensus=cube[:,:,:-1].mean(axis=2)
    domain_rows=[]
    joint=names.index("joint_maximin")
    for j,model in enumerate(cfg["models"]):
        for k,(domain,family) in enumerate(cfg["coordinates"]):
            for m,metric in enumerate(cfg["metrics"]):
                group=noncensus[:,:,j,k,m]
                for index,design in enumerate(names):
                    marginal=interval(group[:,index],POINTWISE)
                    delta=interval(group[:,index]-group[:,joint],POINTWISE)
                    domain_rows.append(dict(site=site,domain=domain,family=family,model=model,metric=metric,
                                            design=design,**marginal,difference_from_joint=delta["mean"],
                                            difference_lower=delta["lower"],difference_upper=delta["upper"]))
    pd.DataFrame(domain_rows).to_csv(folder/"domain_results.csv",index=False)
    key="period=='noncensus' and model=='kriging'"
    eligible=p.query(key).query("design!=@FOCAL")
    best=eligible[np.isclose(eligible["mean"],eligible.groupby("group")["mean"].transform("min"),rtol=1e-13,atol=1e-14)]
    best=best.merge(q.query(key)[["group","design","mean","lower","upper"]],
                    on=["group","design"],suffixes=("_benchmark","_difference"))
    best.to_csv(folder/"strongest.csv",index=False)
    tests=[("treatment","joint_maximin"),("time","contrast_mse"),("primary","sign_error"),
           ("time","joint_maximin"),("primary","joint_maximin")]
    confirm=q.query(key).set_index(["group","design"]).loc[tests].reset_index()
    confirm["upper_below_zero"]=confirm.upper<0
    confirm.to_csv(folder/"decision.csv",index=False)
    print(site,confirm[["group","design","mean","lower","upper"]].to_string(index=False))
    print(best[["group","design","mean_benchmark","mean_difference","lower_difference","upper_difference"]].to_string(index=False))


if __name__ == "__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sites",nargs="+")
    ap.add_argument("--output-dir",type=Path,default=OUT)
    args=ap.parse_args()
    for site in args.sites:
        summarize(site,args.output_dir)
