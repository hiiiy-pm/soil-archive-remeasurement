"""Local predicted gains on the identical paid pilot before two direct-loss rules diverge."""
import argparse
import json
from pathlib import Path

from ranking_run import FAMILIES, OUT, POLICIES, PosteriorState, TargetSet, choose
from ranking_run import local_gains, np, objective_weights, sources
from trials import pd


def audit(site, output):
    loader, baseline, _ = sources(site)
    frame, panels, channels, references, _ = loader(site)
    cfg = json.loads((baseline / "config.json").read_text(encoding="utf-8"))
    targets = [TargetSet(panels,ch) for ch in channels]
    domains = {f:sum(g[2]==f for t in targets for g in t.groups) for f in FAMILIES}
    weights = [np.vstack([objective_weights(t,domains,True),objective_weights(t,domains,False)])
               for t in targets]
    rows = []
    for repeat in range(200):
        with np.load(output/site/f"draw_{repeat:03d}.npz") as draw:
            pilot = draw["pilot"]
            assert tuple(draw["policies"]) == POLICIES
            first = dict(zip(POLICIES,draw["paths"][:,len(pilot)]))
        eligible = np.setdiff1d(np.arange(len(frame)),pilot)
        priority = np.empty(len(frame),int)
        rng = np.random.default_rng(cfg["seed"]+100000+repeat)
        priority[rng.permutation(len(frame))] = np.arange(len(frame))
        gains = np.zeros((4,len(frame)))
        for ch,target,w in zip(channels,targets,weights):
            paid_only = references[panels[ch.panel_ids[0]].name][pilot]
            state = PosteriorState(ch,target,cfg["ratio"],pilot,paid_only,"sign_error")
            ids,values = local_gains(state,w,False)
            gains[:,ids] += values
        maxima = gains[:,eligible].max(axis=1)
        changed = first["split_direct"] != first["pooled_direct"]
        minima = {}
        for design,components in [("split_direct",[0,1,3]),("pooled_direct",[2,3])]:
            choice = choose(eligible,gains[components],priority)
            assert choice == first[design]
            opportunity = np.divide(gains[:,choice],maxima,out=np.ones_like(maxima),where=maxima>0)
            minima[design] = opportunity[[0,1,3]].min()
            rows.append(dict(site=site,repeat=repeat,design=design,
                             record=frame.record_id.iloc[choice],different=changed,
                             time=opportunity[0],treatment=opportunity[1],pooled=opportunity[2],
                             magnitude=opportunity[3],three_objective_minimum=minima[design]))
        assert minima["split_direct"] >= minima["pooled_direct"] - 2e-12
    return rows


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root",type=Path,default=OUT)
    args = parser.parse_args()
    rows = []
    for site in ("tcse","ultuna","lanna"):
        rows.extend(audit(site,args.results_root))
    data = pd.DataFrame(rows)
    data.to_csv(args.results_root/"first_assay.csv",index=False)
    print(data.groupby(["site","design"])[["different","time","treatment","magnitude",
                                           "three_objective_minimum"]].mean().to_string())
