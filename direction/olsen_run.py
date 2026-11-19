"""Select archive records using fixed treatment-contrast Olsen P ranks."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import copy
from fractions import Fraction
import json
from pathlib import Path

from trials import ROOT, TargetSet, exact_targets, np, original, pd
from alert_run import OUT as PARK_RESULTS, SEED as PARK_SEED, warning_path
from joint_run import joint_path, run as run_joint, score_set, tied_best
from ranking_run import allocation, POLICIES as FAMILY_POLICIES
from study import Evaluator, run as run_benchmarks


OUT = ROOT / "direction" / "olsen_results"
NEW_SEED = 326041
GAP_RULES = ("gap_only","gap_joint")


def loader(site):
    if site == "parkgrass":
        from alert_trial import load_trial
    elif site == "hailun":
        from olsen_trial import load_trial
    else:
        from pilot_olsen_trial import load_trial
    return load_trial


def auxiliary_exact(site, panels):
    if site not in {"parkgrass","hailun"}:
        from pilot_olsen_trial import load_auxiliary_exact
        return load_auxiliary_exact(site)
    if site != "parkgrass":
        from olsen_trial import load_auxiliary_exact
        return load_auxiliary_exact(site)
    from alert_trial import SOURCE
    data = pd.read_csv(SOURCE / "paired_soils.csv", dtype=str, keep_default_na=False)
    return {panels[0].name: dict(enumerate(map(Fraction, data.olsen_p_mg_kg)))}


def midranks(gaps):
    counts = Counter(gaps)
    ranks, smaller = {}, 0
    for value in sorted(counts):
        n = counts[value]
        ranks[value] = smaller+(n+1)/2
        smaller += n
    return np.array([ranks[v] for v in gaps])


def freeze_gaps(site, frame, panels, channels, targets):
    auxiliary = auxiliary_exact(site, panels)
    domain_count = sum(group[2]=="treatment" for t in targets for group in t.groups)
    assert domain_count > 0
    inputs, weighted, record_score = [], [], np.zeros(len(frame))
    for channel, target in zip(channels,targets):
        gaps = exact_targets(panels,target,auxiliary)
        modified = copy(target)
        modified.primary_weights = target.primary_weights.copy()
        for pi, domain, family, start, end, _ in target.groups:
            if family != "treatment":
                continue
            abs_gap = list(map(abs,gaps[start:end]))
            ranks = midranks(abs_gap)
            n = len(abs_gap)
            mult = 2*(n+1-ranks)/(n+1)
            assert mult.min()>0 and np.isclose(mult.mean(),1)
            modified.primary_weights[start:end] *= mult
            record_score[channel.rows] += (mult/(domain_count*n)) @ (target.C[start:end]**2)
            for j in range(n):
                sample_ids, coef = target.definitions[start+j][1:]
                labels = [(panels[pi].cells[int(c)],int(v)) for c,v in zip(sample_ids,coef)]
                inputs.append(dict(domain=domain,contrast=j+1,cells=str(labels),gap_exact=str(abs_gap[j]),
                                   gap=float(abs_gap[j]),midrank=ranks[j],multiplier=mult[j]))
        weighted.append(modified)
    return weighted,record_score,pd.DataFrame(inputs)


def archive_path(pilot, scores, seed):
    n = len(scores)
    priority = np.empty(n,int)
    priority[np.random.default_rng(seed).permutation(n)] = np.arange(n)
    eligible = np.ones(n,bool)
    eligible[pilot] = False
    path = list(pilot)
    while eligible.any():
        ids = np.flatnonzero(eligible)
        tied = tied_best(ids,scores[ids])
        chosen = tied[np.argmin(priority[tied])]
        path.append(chosen)
        eligible[chosen] = False
    return np.array(path,dtype=np.int16)


def comparison_root(site,out):
    return PARK_RESULTS if site=="parkgrass" else out


def build_inputs(site,out,trial_seed=NEW_SEED):
    get_trial = loader(site)
    if site != "parkgrass":
        run_benchmarks(site,0,0,output_root=out/"benchmarks",loader=get_trial,trial_seed=trial_seed)
    parent = comparison_root(site,out)
    cfg=json.loads((parent/"benchmarks"/site/"config.json").read_text())
    frame,panels,channels,_,_=get_trial(site)
    targets=[TargetSet(panels,ch) for ch in channels]
    _,priority,rows=freeze_gaps(site,frame,panels,channels,targets)
    folder=out/site
    (folder/"draws").mkdir(parents=True,exist_ok=True)
    rows.to_csv(folder/"frozen_gaps.csv",index=False)
    pd.DataFrame({"record_id":frame.record_id,"static_priority":priority}).to_csv(folder/"record_priorities.csv",index=False)
    cfg["previous_designs"]=cfg["designs"]+["joint_maximin","alternate",*FAMILY_POLICIES,"pilot_warning"]
    cfg["designs"]=cfg["previous_designs"]+list(GAP_RULES)
    assert len(set(cfg["designs"]))==26 and cfg["seed"]==(PARK_SEED if site=="parkgrass" else trial_seed)
    (folder/"config.json").write_text(json.dumps(cfg,indent=2)+"\n",encoding="utf-8")


def run_chunk(site,start,stop,out,trial_seed=NEW_SEED):
    get_trial = loader(site)
    parent=comparison_root(site,out)
    if site != "parkgrass":
        run_benchmarks(site,start,stop,output_root=parent/"benchmarks",loader=get_trial,trial_seed=trial_seed)
        run_joint(site,start,stop,output_root=parent/"fixed",baseline_root=parent/"benchmarks",loader=get_trial)
    base=parent/"benchmarks"/site
    cfg=json.loads((base/"config.json").read_text())
    frame,panels,channels,responses,exact=get_trial(site)
    size,ratio=len(frame),cfg["ratio"]
    targets=[TargetSet(panels,ch) for ch in channels]
    weighted,priority,_=freeze_gaps(site,frame,panels,channels,targets)
    covs=[ch.kernel+ratio*np.eye(len(ch.rows)) for ch in channels]
    evaluator=Evaluator(panels,channels,targets,exact)
    for rep in range(start,stop):
        name=f"draw_{rep:03d}.npz"
        dest=out/site/"draws"/name
        if dest.exists():
            continue
        seed=cfg["seed"]+rep
        pilot=np.random.default_rng(seed).choice(size,cfg["budgets"][0],replace=False)
        with np.load(base/name) as reference:
            np.testing.assert_array_equal(reference["pilot"],pilot)
            pilot_score=reference["scores"][0,0]
        tie_seed=seed+100000
        oracle=original.PaidOracle(responses)
        first=oracle.reveal(pilot)
        focal=joint_path(size,panels,channels,weighted,ratio,pilot,first,oracle.reveal,"maximin",tie_seed)
        assert len(oracle.paid)==size
        paths=[archive_path(pilot,priority,tie_seed),focal]
        names=list(GAP_RULES)
        if site!="parkgrass":
            preceding=[allocation(size,panels,channels,targets,ratio,pilot,original.PaidOracle(responses),policy,tie_seed)
                       for policy in FAMILY_POLICIES]
            warn,_,_=warning_path(size,panels,channels,targets,ratio,pilot,original.PaidOracle(responses),tie_seed)
            paths=[*preceding,warn,*paths]
            names=[*FAMILY_POLICIES,"pilot_warning",*names]
        settings={p.name:original.pilot_settings(p,pilot,first[p.name]) for p in panels}
        scores=[]
        for path in paths:
            np.testing.assert_array_equal(np.sort(path),np.arange(size))
            np.testing.assert_array_equal(path[:len(pilot)],pilot)
            scored=np.stack([score_set(path[:b],panels,channels,targets,covs,settings,responses,evaluator,size)
                             for b in cfg["budgets"]])
            np.testing.assert_allclose(scored[0],pilot_score,rtol=0,atol=0)
            assert np.isfinite(scored).all() and not scored[-1].any()
            scores.append(scored)
        tmp=dest.with_suffix(".tmp")
        with tmp.open("wb") as stream:
            np.savez_compressed(stream,pilot=pilot,policies=names,paths=paths,scores=scores)
        tmp.replace(dest)
    return site,start,stop


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("site")
    ap.add_argument("--start",type=int,default=0)
    ap.add_argument("--stop",type=int,default=200)
    ap.add_argument("--workers",type=int,default=1)
    ap.add_argument("--output-dir",type=Path,default=OUT)
    args=ap.parse_args()
    if not 0<=args.start<args.stop<=200 or args.workers<1:
        ap.error("Enter a nonempty subset of the 200 pilots and a positive worker count.")
    build_inputs(args.site,args.output_dir)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(run_chunk,args.site,i,min(i+4,args.stop),args.output_dir) for i in range(args.start,args.stop,4)]
        for job in as_completed(jobs):
            print(*job.result(),flush=True)
