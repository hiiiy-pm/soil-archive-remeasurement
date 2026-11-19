"""Diagnose the signed comparisons of the original Lanna allocations on their paid paths."""
import argparse
import ast
import json
from pathlib import Path
from statistics import NormalDist

from multiperiod_run import OUT as EXPERIMENT
from multiperiod_trial import SITE, load_trial
from trials import TargetSet, exact_targets, np, pd
from experiment import conditional
from acquisition import PosteriorState, sign_gain
from joint_run import choose


FOLDER = Path(__file__).resolve().parent / "ranking_diagnosis"
RULES = ["joint_maximin", "alternate", "sign_error", "sign_variance", "contrast_mse"]
PHASES = [[0], [1, 2], [3], [4], [5]]


def audit_contrasts(output):
    frame, panels, (ch,), responses, exact = load_trial()
    baseline = EXPERIMENT/"benchmarks"/SITE
    cfg = json.loads((baseline/"config.json").read_text())
    coordinates = cfg["coordinates"]
    target = TargetSet(panels, ch)
    keep = target.primary
    c = target.C[keep]
    actual = np.array([float(v) for v in exact_targets(panels, target, exact)])[keep]
    reference = responses[panels[0].name]
    cov = ch.kernel + cfg["ratio"]*np.eye(len(frame))
    inventory = pd.read_csv(baseline/"contrast_inventory.csv")
    info = inventory[inventory.family.isin(["time", "treatment"])].copy().reset_index(drop=True)
    np.testing.assert_allclose(info.reference, actual)
    assert np.all(actual != 0)
    a,b = [],[]
    for row in info.itertuples():
        cells = ast.literal_eval(row.cells)
        a.append(next(cell for cell,w in cells if w==1))
        b.append(next(cell for cell,w in cells if w==-1))
    info["first_year"],info["first_treatment"] = zip(*a)
    info["second_year"],info["second_treatment"] = zip(*b)
    info["treatment_pair"] = info.first_treatment+" minus "+info.second_treatment
    info["weight_in_family"] = (1/2)/info.groupby(["domain","family"]).contrast.transform("size")
    d = len(frame)
    periods = cfg["budgets"][:-1]
    repeats = cfg["replicates"]
    ids = [cfg["designs"].index(name) for name in RULES[2:]]
    all_rates = np.empty((repeats,len(RULES),len(periods),len(actual)))
    all_mse = np.zeros((len(RULES),len(periods),len(actual)))
    all_resolved = np.zeros_like(all_mse)
    discordance = np.zeros((2, len(actual)))
    for rep in range(repeats):
        filename=f"draw_{rep:03d}.npz"
        with np.load(baseline/filename) as base, np.load(EXPERIMENT/"fixed"/SITE/filename) as fix:
            np.testing.assert_array_equal(base["pilot"],fix["pilot"])
            masks=np.unpackbits(base["selections"][ids,:-1],axis=-1,count=d).astype(bool)
            paths=np.empty((3,len(periods),d),bool)
            for n,path in enumerate(fix["paths"]):
                paths[n]=[np.isin(np.arange(d),path[:budget]) for budget in periods]
            saved=np.concatenate([fix["scores"][:,:-1,0],base["scores"][ids,:-1,0]])
        schedules=np.concatenate([paths,masks],axis=0)
        cache={}
        errors=np.zeros((6,len(periods),len(actual),3))
        for pi,schedule in enumerate(schedules):
            for bi,paid in enumerate(schedule):
                key=paid.tobytes()
                assert paid.sum()==periods[bi]
                if key not in cache:
                    ix=np.flatnonzero(paid)
                    pred=ch.x+conditional(cov,ix,reference[ix]-ch.x[ix])
                    pred[ix]=reference[ix]
                    est=(c@np.column_stack([pred,pred]))[:,0]
                    resolved=~c[:,~paid].any(axis=1)
                    est[resolved]=actual[resolved]
                    cache[key]=np.column_stack((np.sign(est)!=np.sign(actual),(est-actual)**2,resolved))
                errors[pi,bi]=cache[key]
        # Every one of the 200 replicates, six trajectories and nine budgets
        # must reconstruct the originally scored domain errors from the individual differences.
        for (domain,family),group in info.groupby(["domain","family"]):
            ci=coordinates.index([domain,family])
            reconstructed=errors[:,:,group.index,:].mean(axis=2)
            for k,metric in enumerate(["direction","native_mse"]):
                np.testing.assert_allclose(reconstructed[:,:,k],saved[:,:,ci,cfg["metrics"].index(metric)],atol=1e-9)
        for ri,phases in enumerate(PHASES):
            arr=errors[phases].mean(axis=0)
            all_rates[rep,ri]=arr[:,:,0]
            all_mse[ri]+=arr[:,:,1]/repeats
            all_resolved[ri]+=arr[:,:,2]/repeats
        diff=errors[0,:,:,0]-errors[3,:,:,0]
        discordance[0]+=(diff==1).sum(axis=0)
        discordance[1]+=(diff==-1).sum(axis=0)
    rates=all_rates.mean(axis=2)
    means=rates.mean(axis=0)
    for i,rule in enumerate(RULES):
        info[rule+"_error"]=means[i]
        info[rule+"_native_rmse"]=np.sqrt(all_mse[i].mean(axis=0))
        info[rule+"_resolved"]=all_resolved[i].mean(axis=0)
    direct=RULES.index("sign_error")
    proxy=RULES.index("sign_variance")
    info["variance_minus_direct"]=means[proxy]-means[direct]
    info["joint_minus_variance"]=means[0]-means[proxy]
    info["joint_minus_direct"]=means[0]-means[direct]
    info["contribution_joint_minus_direct"]=info.weight_in_family*info.joint_minus_direct
    info["joint_wrong_direct_right"]=discordance[0].astype(int)
    info["joint_right_direct_wrong"]=discordance[1].astype(int)
    # The experimental unit for Monte Carlo precision is one common pilot.
    delta=rates[:,0]-rates[:,direct]
    se=delta.std(axis=0,ddof=1)/np.sqrt(repeats)
    for family,g in info.groupby("family"):
        z=NormalDist().inv_cdf(1-0.05/(2*len(g)))
        info.loc[g.index,"simultaneous_lower"]=info.loc[g.index,"joint_minus_direct"]-z*se[g.index]
        info.loc[g.index,"simultaneous_upper"]=info.loc[g.index,"joint_minus_direct"]+z*se[g.index]
    output.mkdir(exist_ok=True,parents=True)
    info.to_csv(output/"contrast_losses.csv",index=False)
    group_rows=[]
    subset=info[info.family=="treatment"]
    for level,frame_group in [("all",[("all",subset)]),("field",subset.groupby("domain")),
                              ("pair",subset.groupby("treatment_pair")),("small",subset.groupby("small")),
                              ("year",subset.groupby("first_year"))]:
        for label,rows in frame_group:
            idx=rows.index
            w=rows.weight_in_family.to_numpy()
            row=dict(level=level,category=label,contrasts=len(rows),weight=w.sum(),
                     joint_wrong_direct_right=rows.joint_wrong_direct_right.sum(),
                     joint_right_direct_wrong=rows.joint_right_direct_wrong.sum())
            for n,rule in enumerate(RULES):
                row[rule+"_error"]=np.average(means[n,idx],weights=w)
            for name,diff in [("joint_minus_direct",rates[:,0]-rates[:,direct]),
                              ("variance_minus_direct",rates[:,proxy]-rates[:,direct]),
                              ("joint_minus_variance",rates[:,0]-rates[:,proxy])]:
                draw=diff[:,idx]@w
                row[name+"_contribution"]=draw.mean()
                se=draw.std(ddof=1)/np.sqrt(repeats)
                row[name+"_mcse"]=se
            group_rows.append(row)
    pd.DataFrame(group_rows).to_csv(output/"treatment_attribution.csv",index=False)
    pooled=[]
    for fam in ["time","treatment"]:
        group=info[info.family.eq(fam)]
        w=group.weight_in_family.to_numpy()
        for n,rule in enumerate(RULES):
            pooled.append((fam,rule,np.average(means[n,group.index],weights=w)))
    pd.DataFrame(pooled,columns=["family","design","direction"]).to_csv(output/"families.csv",index=False)
    return info


def audit_decisions(output):
    """Compare eligible next assays at each step of the archived joint selection paths."""
    frame, panels, (ch,),responses,_=load_trial()
    cfg=json.loads((EXPERIMENT/"benchmarks"/SITE/"config.json").read_text())
    target=TargetSet(panels,ch)
    fam=np.full(len(target.C),"",dtype=object)
    for _,_,label,first,last,_ in target.groups:
        fam[first:last]=label
    fam=fam[target.primary]
    assert set(fam)=={"time","treatment"}
    y=responses[panels[0].name]
    n=len(frame)
    records=[]
    gains_names=["treatment_risk","time_risk","pooled_risk","pooled_variance","magnitude"]
    for rep in range(cfg["replicates"]):
        with np.load(EXPERIMENT/"fixed"/SITE/f"draw_{rep:03d}.npz") as f:
            pilot,path=f["pilot"],f["paths"][0]
        with np.load(EXPERIMENT/"benchmarks"/SITE/f"draw_{rep:03d}.npz") as archived:
            next_actual={key:int(archived["path_"+policy][len(pilot)])
                         for key,policy in [("pooled_risk","sign_error"),("pooled_variance","sign_variance"),
                                            ("magnitude","contrast_mse")]}
        var=PosteriorState(ch,target,cfg["ratio"],pilot,y[pilot],"sign_variance")
        mag=PosteriorState(ch,target,cfg["ratio"],pilot,y[pilot],"contrast_mse")
        priority=np.empty(n,int)
        priority[np.random.default_rng(cfg["seed"]+100000+rep).permutation(n)]=np.arange(n)
        for prior,actual in enumerate(path[len(pilot):],len(pilot)):
            remaining=var.unobserved>0
            ids, variance_gain=var.gains()
            _, mse_gain=mag.gains()
            den=var.R.diagonal()[ids]
            v=var.var[remaining]
            z=np.abs(var.mu[remaining])/np.sqrt(var.scale*v)
            rho=np.abs(var.cross[np.ix_(remaining,ids)])/np.sqrt(v[:,None]*den)
            risk=sign_gain(z[:,None],rho)
            gains=np.zeros((5,n))
            for j,which in enumerate(["treatment","time"]):
                parts=(fam[remaining]==which)
                # Each family has half the acquisition weight; double it to
                # express the separate gain on that family's usual scale.
                gains[j,ids]=2*var.weights[remaining][parts]@risk[parts]
            gains[2,ids]=var.weights[remaining]@risk
            np.testing.assert_allclose(gains[2,ids],(gains[0,ids]+gains[1,ids])/2,atol=1e-13)
            gains[3,ids]=variance_gain
            gains[4,ids]=mse_gain
            assert choose(ids,gains[3:],priority)==actual
            row=dict(pilot=rep,paid_before=prior,remaining=len(ids),joint=int(actual))
            for j,label in enumerate(gains_names):
                idx=choose(ids,gains,priority,component=j)
                row[label+"_pick"]=idx
                row["joint_"+label]=gains[j,actual]
                row["best_"+label]=gains[j,idx]
            if prior==len(pilot):
                for key,idx in next_actual.items():
                    assert row[key+"_pick"]==idx
            for pick,prefix in [("pooled_risk_pick","pooled_choice"),
                                ("pooled_variance_pick","variance_choice"),
                                ("treatment_risk_pick","treatment_choice")]:
                for j,key in enumerate(gains_names):
                    row[prefix+"_"+key]=gains[j,row[pick]]
            records.append(row)
            var.update(actual,y[actual])
            mag.update(actual,y[actual])
    steps=pd.DataFrame(records)
    steps.to_csv(output/"joint_next_assay.csv",index=False)
    z=steps[steps.paid_before==cfg["budgets"][0]]
    def lower(a,b):
        return (a<b)&~np.isclose(a,b,rtol=1e-12,atol=1e-15)
    for label,f in [("common pilot",z),("all nonforced joint-path states",steps[steps.remaining>1])]:
        weak=lower(f.joint_treatment_risk,f.pooled_choice_treatment_risk)
        more_mag=lower(f.pooled_choice_magnitude,f.joint_magnitude)
        pooled_trade=lower(f.pooled_choice_treatment_risk,f.best_treatment_risk)
        print(label,len(f),"pooled risk below treatment-only",int(pooled_trade.sum()),
              "joint below pooled on treatment",int(weak.sum()),
              "joint below pooled on treatment and higher in magnitude",int((more_mag&weak).sum()))


def diagnosis_table(output):
    group=pd.read_csv(output/"treatment_attribution.csv").set_index(["level","category"])
    items=pd.read_csv(output/"contrast_losses.csv")
    steps=pd.read_csv(output/"joint_next_assay.csv")
    pilots=steps[steps.paid_before==steps.paid_before.min()]
    lines=[r"\begin{table}[!htbp]\centering\small",
           r"\caption{Sources of treatment-ranking error at Lanna. Sign errors average nine non-census budgets within 200 shared pilots. The contribution is the change in the overall treatment error rate (percentage points) attributable to each row with equal weight for the two field halves. Field, treatment pair and contrast size provide three breakdowns of the same 34 contrasts.}\label{tab:lanna-diagnosis}",
           r"\setlength{\tabcolsep}{3.3pt}\begin{tabular}{@{}lrrrrrr@{}}\toprule",
           r"& & & \multicolumn{3}{c}{Wrong signs (\%)}& Joint minus direct\\",
           r"\cmidrule(lr){4-6}",
           r"Group & $n$ & Weight (\%) & Direct & Variance & Joint & Contribution (pp)\\\midrule"]
    def block(keys):
        for level,category,label in keys:
            r=group.loc[(level,category)]
            vals=[str(int(r.contrasts)),f"{100*r.weight:.2f}",f"{100*r.sign_error_error:.2f}",
                  f"{100*r.sign_variance_error:.2f}",f"{100*r.joint_maximin_error:.2f}",
                  f"{100*r.joint_minus_direct_contribution:+.3f}"]
            lines.append(" & ".join([label,*vals])+r"\\")
    block([("all","all","All treatment contrasts"),("field","P_High","High field half"),
           ("field","P_Low","Low field half")])
    lines.append(r"\midrule")
    block([("pair","Control minus P","No P vs annual P"),
           ("pair","Control minus P6","No P vs six-year P"),
           ("pair","P minus P6","Annual P vs six-year P")])
    lines.append(r"\midrule")
    block([("small","True","Smallest within-field quarter"),
           ("small","False","Remaining contrasts")])
    worst=items[items.family.eq("treatment")].sort_values("contribution_joint_minus_direct").iloc[-1]
    assert (worst.domain,worst.first_year,worst.treatment_pair)==("P_High",2001,"P minus P6")
    def lower(a,b):
        return (a<b)&~np.isclose(a,b,rtol=1e-12,atol=1e-15)
    weak=lower(pilots.joint_treatment_risk,pilots.pooled_choice_treatment_risk)
    larger=lower(pilots.pooled_choice_magnitude,pilots.joint_magnitude)
    spend=lower(pilots.pooled_choice_treatment_risk,pilots.best_treatment_risk)
    lines.extend([r"\bottomrule\end{tabular}\par\smallskip",
        r"\begin{minipage}{.97\linewidth}\footnotesize",
        f"The largest positive weighted contribution came from annual P minus six-year P in the High half in 2001 "
        f"({worst.reference:.2f}~mg P kg$^{{-1}}$). Its sign error was {100*worst.joint_maximin_error:.2f}\\% "
        f"under joint selection and {100*worst.sign_error_error:.2f}\\% under direct sign selection. Both laboratory "
        f"values were paid for in {100*worst.joint_maximin_resolved:.2f}\\% and {100*worst.sign_error_resolved:.2f}\\% "
        f"of the matched budget--pilot settings, respectively. Direct and variance selection each target both sign families.",
        r"\par\smallskip",
        f"In {spend.sum()} of {len(pilots)} common pilots, the next assay chosen for pooled direct sign risk gave less "
        "predicted treatment sign-risk reduction than the assay maximizing that gain alone. "
        f"In {weak.sum()} pilots the joint choice gave less predicted treatment sign-risk reduction than the pooled direct "
        f"choice. In {(weak&larger).sum()} of these it gave more predicted magnitude-error reduction. "
        "These are one-assay forecasts at the same paid information state.",
        r"\end{minipage}\end{table}"])
    return "\n".join(lines)+"\n"


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=FOLDER)
    parser.add_argument("--table",type=Path,help="Optionally typeset the treatment attribution and first-assay comparison.")
    args=parser.parse_args()
    audit_contrasts(args.output)
    audit_decisions(args.output)
    if args.table:
        args.table.write_text(diagnosis_table(args.output),encoding="utf8")
