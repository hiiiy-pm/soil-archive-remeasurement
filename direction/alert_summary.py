"""Evaluate pilot discrimination separately from a paid allocation response."""
import json
from statistics import NormalDist

from scipy.stats import rankdata
from alert_run import OUT, SITE
from ranking_run import POLICIES
from joint_summary import domain_family_weights
from trials import np, pd


ENDPOINTS = [("time","direction"),("treatment","direction"),("primary","scaled_mse")]
CRITICAL = NormalDist().inv_cdf(1-.05/(2*70))


def interval(draws, critical=CRITICAL):
    v = np.asarray(draws)
    v = v[np.isfinite(v)]
    if len(v)<2:
        return dict(n=len(v),mean=float(v[0]) if len(v) else np.nan,lower=np.nan,upper=np.nan)
    m = v.mean()
    d = critical*v.std(ddof=1)/np.sqrt(len(v))
    return dict(n=len(v),mean=m,lower=m-d,upper=m+d)


def auc(ranks, events):
    n1 = events.sum()
    n0 = len(events)-n1
    return (ranks[events].sum()-n1*(n1+1)/2)/(n1*n0) if n1 and n0 else np.nan


def analyze():
    cfg = json.loads((OUT/"benchmarks"/SITE/"config.json").read_text())
    names = cfg["designs"]+["joint_maximin","alternate",*POLICIES,"pilot_warning"]
    assert len(names)==len(set(names))==24
    raw, diagnostics = [], []
    for i in range(200):
        file = f"draw_{i:03d}.npz"
        with (np.load(OUT/"benchmarks"/SITE/file) as base, np.load(OUT/"fixed"/SITE/file) as joint,
              np.load(OUT/"warning"/file) as extra):
            pilot = np.random.default_rng(cfg["seed"]+i).choice(cfg["records"],cfg["budgets"][0],replace=False)
            for dr in (base,joint,extra):
                np.testing.assert_array_equal(dr["pilot"],pilot)
            bits = np.unpackbits(base["selections"],axis=-1,count=cfg["records"])
            np.testing.assert_array_equal(bits.sum(-1),np.broadcast_to(cfg["budgets"],bits.shape[:2]))
            joint_scores = joint["scores"]
            saved = extra["scores"]
            assert list(extra["policies"])==[*POLICIES,"pilot_warning"]
            block=np.concatenate((base["scores"],joint_scores[0:1],joint_scores[1:3].mean(axis=0,keepdims=True),saved))
            assert np.isfinite(block).all() and not block[:,-1].any()
            raw.append(block)
            use = extra["initially_unresolved"] & extra["reference_nonzero"]
            q = extra["risk"][use]
            hi = q>np.median(extra["risk"][extra["initially_unresolved"]])
            ranks = rankdata(q)
            changed = int(joint["paths"][0,len(pilot)] != extra["paths"][-1,len(pilot)])
            for b,errors in zip(cfg["budgets"][:-1],extra["joint_wrong"][:-1, use]):
                diagnostics.append(dict(pilot=i,budget=b,initially_eligible=len(q),events=int(errors.sum()),
                                        auc=auc(ranks,errors),high_n=int(hi.sum()),low_n=int((~hi).sum()),
                                        high_error=errors[hi].mean() if hi.any() else np.nan,
                                        low_error=errors[~hi].mean() if (~hi).any() else np.nan,first_changed=changed))
    d=pd.DataFrame(diagnostics)
    d.to_csv(OUT/"warning_by_pilot_budget.csv",index=False)
    summary=[]
    for name,g in [("noncensus",d),("pilot",d[d.budget==cfg["budgets"][0]])]:
        for metric in ("auc","high_error","low_error"):
            v=g.groupby("pilot")[metric].mean().dropna()
            summary.append(dict(period=name,metric=metric,defined_budget_draws=g[metric].count(),
                                total_budget_draws=len(g),**interval(v,CRITICAL if metric=="auc" and name=="noncensus" else 1.95996398454)))
    pd.DataFrame(summary).to_csv(OUT/"warning_discrimination.csv",index=False)
    cube=np.array(raw)
    rows,paired = [],[]
    for idx,model in enumerate(cfg["models"]):
        for group,metric in ENDPOINTS:
            k=cfg["metrics"].index(metric)
            w=domain_family_weights(cfg["coordinates"],group)
            scores=cube[:,:,:,idx,:,k]@w
            periods=[("noncensus",scores[:,:,:-1].mean(axis=2))]+[
                (str(b),scores[:,:,j]) for j,b in enumerate(cfg["budgets"])]
            for period,values in periods:
                for j,name in enumerate(names):
                    key=dict(model=model,period=period,group=group,metric=metric,design=name)
                    rows.append(dict(**key,**interval(values[:,j],1.95996398454)))
                    if name!=names[-1]:
                        crit=CRITICAL if model=="kriging" and period=="noncensus" else 1.95996398454
                        paired.append(dict(**key,**interval(values[:,-1]-values[:,j],crit)))
            if model=="kriging":
                comparator="contrast_mse" if group=="time" else "sign_error"
                diff = scores[:,names.index("joint_maximin"),:-1].mean(-1)-scores[:,names.index(comparator),:-1].mean(-1)
                pd.DataFrame({"pilot":np.arange(200),"difference":diff}).to_csv(OUT/("original_"+group+".csv"),index=False)
    pd.DataFrame(rows).to_csv(OUT/"performance.csv",index=False)
    pd.DataFrame(paired).to_csv(OUT/"warning_minus_comparator.csv",index=False)


if __name__=="__main__":
    analyze()
