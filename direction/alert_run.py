"""Independent field replay of the frozen pilot warning and its one-assay response."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json

from trials import ROOT, TargetSet, original, np
from experiment import conditional
from acquisition import PosteriorState, sign_gain
from scipy.special import ndtr
from joint_run import choose, score_set, run as run_joint
from ranking_run import allocation, POLICIES, objective_weights
from study import Evaluator, run as run_benchmarks


OUT = ROOT / "direction" / "alert_results"
SEED = 196031
SITE = "parkgrass"


def warning_path(size, panels, channels, targets, ratio, pilot, oracle, seed):
    paid = oracle.reveal(pilot)
    states = [PosteriorState(ch, t, ratio, pilot, paid[panels[ch.panel_ids[0]].name], "sign_variance")
              for ch, t in zip(channels, targets)]
    magnitude = [PosteriorState(ch, t, ratio, pilot, paid[panels[ch.panel_ids[0]].name], "contrast_mse")
                 for ch, t in zip(channels, targets)]
    counts = {family: sum(g[2] == family for t in targets for g in t.groups)
              for family in ("treatment", "time")}
    first_gain = np.zeros(size)
    risk, initially_unresolved = [], []
    for state, t in zip(states, targets):
        w = objective_weights(t, counts, True)[1]
        is_treatment = w > 0
        active = (state.unobserved > 0) & is_treatment
        q = np.zeros(len(w))
        z = abs(state.mu[active]) / np.sqrt(state.scale * state.var[active])
        q[active] = ndtr(-z)
        risk.extend(q[is_treatment])
        initially_unresolved.extend(active[is_treatment])
        ids = np.flatnonzero(state.available)
        den = state.R.diagonal()[ids]
        r = abs(state.cross[np.ix_(active, ids)]) / np.sqrt(state.var[active,None] * den)
        first_gain[state.channel.rows[ids]] += (w[active] * q[active]) @ sign_gain(z[:,None], r)
    priority = np.empty(size, int)
    priority[np.random.default_rng(seed).permutation(size)] = np.arange(size)
    available = np.ones(size, bool)
    available[pilot] = False
    lookup = [{row: i for i, row in enumerate(ch.rows)} for ch in channels]
    path = list(pilot)
    while available.any():
        ids = np.flatnonzero(available)
        if len(path) == len(pilot) and first_gain[ids].max() > 0:
            choice = choose(ids, first_gain[None,:], priority, component=0)
        else:
            gains = np.zeros((2, size))
            for k, collection in enumerate((states, magnitude)):
                for state in collection:
                    rows, gain = state.gains()
                    gains[k, rows] += gain
            choice = choose(ids, gains, priority)
        measured = oracle.reveal(np.array([choice]))
        for j, ch in enumerate(channels):
            if choice in lookup[j]:
                y = measured[panels[ch.panel_ids[0]].name][0]
                for state in (states[j], magnitude[j]):
                    state.update(lookup[j][choice], y)
        path.append(choice)
        available[choice] = False
    assert len(oracle.paid) == size
    return np.array(path, dtype=np.int16), np.array(risk), np.array(initially_unresolved)


def treatment_errors(path, budgets, panels, channels, targets, covs, responses, evaluator):
    """After allocation, score every treatment comparison against exact archived signs."""
    wrong, nonzero = [], []
    for t, (_, signs, _) in zip(targets, evaluator.parts):
        treat = np.zeros(len(signs), bool)
        for _, _, family, a, b, _ in t.groups:
            if family == "treatment":
                treat[a:b] = True
        nonzero.extend(signs[treat] != 0)
    for b in budgets:
        errors = []
        for ch,t,cov,(_,signs,_) in zip(channels,targets,covs,evaluator.parts):
            mask = np.isin(ch.rows, path[:b])
            ix = np.flatnonzero(mask)
            y = responses[panels[ch.panel_ids[0]].name][ch.rows[ix]]
            pred = ch.x + conditional(cov, ix, y-ch.x[ix])
            pred[ix] = y
            for _,_,family,a,end,_ in t.groups:
                if family == "treatment":
                    c = t.C[a:end]
                    e = np.sign(c@pred) != signs[a:end]
                    e[~np.any(c[:,~mask] != 0, axis=1)] = False
                    errors.extend(e)
        wrong.append(errors)
    return np.array(wrong), np.array(nonzero)


def chunk(start, stop, out):
    from alert_trial import load_trial
    run_benchmarks(SITE, start, stop, output_root=out/"benchmarks", loader=load_trial, trial_seed=SEED)
    run_joint(SITE, start, stop, output_root=out/"fixed", baseline_root=out/"benchmarks", loader=load_trial)
    cfg = json.loads((out/"benchmarks"/SITE/"config.json").read_text())
    frame, panels, channels, responses, exact = load_trial(SITE)
    n = len(frame)
    targets = [TargetSet(panels,ch) for ch in channels]
    evaluator = Evaluator(panels,channels,targets,exact)
    covs = [c.kernel + cfg["ratio"]*np.eye(len(c.rows)) for c in channels]
    loc = out/"warning"
    loc.mkdir(exist_ok=True)
    for i in range(start,stop):
        filename=f"draw_{i:03d}.npz"
        dest = loc/filename
        if dest.exists():
            continue
        with np.load(out/"fixed"/SITE/filename) as ref:
            pilot = ref["pilot"]
            joint = ref["paths"][0]
            shared_initial = ref["scores"][0,0]
        seed = SEED+i+100000
        settings={p.name:original.pilot_settings(p,pilot,responses[p.name][pilot]) for p in panels}
        path,q,unresolved = warning_path(n,panels,channels,targets,cfg["ratio"],pilot,
                                         original.PaidOracle(responses),seed)
        errors,nonzero = treatment_errors(joint,cfg["budgets"],panels,channels,targets,covs,responses,evaluator)
        assert len(q)==errors.shape[1] and len(unresolved)==len(nonzero)
        paths = [allocation(n,panels,channels,targets,cfg["ratio"],pilot,original.PaidOracle(responses),rule,seed)
                 for rule in POLICIES] + [path]
        scores=[]
        for p in paths:
            np.testing.assert_array_equal(np.sort(p),np.arange(n))
            np.testing.assert_array_equal(p[:len(pilot)],pilot)
            scores.append([score_set(p[:b],panels,channels,targets,covs,settings,responses,evaluator,n)
                           for b in cfg["budgets"]])
            np.testing.assert_array_equal(scores[-1][0],shared_initial)
        scores=np.asarray(scores)
        assert np.isfinite(scores).all() and not scores[:,-1].any() and not errors[-1].any()
        temp=dest.with_suffix(".tmp")
        with temp.open("wb") as stream:
            np.savez_compressed(stream,pilot=pilot,scores=scores,paths=paths,risk=q,
                                initially_unresolved=unresolved,reference_nonzero=nonzero,
                                joint_wrong=errors,policies=[*POLICIES,"pilot_warning"])
        temp.replace(dest)
    return start,stop


if __name__ == "__main__":
    from alert_trial import load_trial
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers",type=int,default=1)
    ap.add_argument("--start",type=int,default=0)
    ap.add_argument("--stop",type=int,default=200)
    ap.add_argument("--output-dir",type=Path,default=OUT)
    args=ap.parse_args()
    assert 0<=args.start<args.stop<=200 and args.workers>0
    run_benchmarks(SITE,0,0,output_root=args.output_dir/"benchmarks",loader=load_trial,trial_seed=SEED)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(chunk,i,min(i+2,args.stop),args.output_dir) for i in range(args.start,args.stop,2)]
        for job in as_completed(jobs):
            print("Completed pilots",*job.result(),flush=True)
