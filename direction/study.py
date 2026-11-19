"""Reproducible paid-pilot tests of contrast sign acquisition on two public trials."""
from trials import ROOT, TargetSet, exact_targets, load_trial, np, pd, original, discrepancy_kernel
from acquisition import MODEL_RULES, select_path
from experiment import conditional
from comparators import matched_adaptive
from target_cells import metadata_labels
import argparse
import json
from pathlib import Path

OUT = ROOT/'direction'/'results'
SEEDS = {'tcse':712341,'ultuna':815237}
PUBLIC_RULES = ['random','cell_balance','target_cell_balance','maximin','active_sampling',
                'retained_contrast','conversion','concentration',
                'a2_if_year','a4_loo_year','a4_loo_cell','a4_loo_targetcell']
RULES = MODEL_RULES+PUBLIC_RULES
METRICS = ['direction','scaled_mse','three_class','small_direction','native_mse']
MODELS = ['kriging','ridge']

class Evaluator:
    def __init__(self,panels,channels,targets,fractions):
        self.parts = []
        self.inventory = []
        for ch,target in zip(channels,targets):
            exact = exact_targets(panels,target,fractions)
            values = np.array([float(v) for v in exact])
            signs = np.array([int(v>0)-int(v<0) for v in exact])
            small = np.zeros(len(exact),bool)
            for pi,domain,family,start,end,scale in target.groups:
                nonzero = [abs(v) for v in exact[start:end] if v]
                cutoff = sorted(nonzero)[int(np.ceil(len(nonzero)*.25))-1] if nonzero else 0
                small[start:end] = [0<abs(v)<=cutoff for v in exact[start:end]]
                for j in range(start,end):
                    _,ids,coef = target.definitions[j]
                    cells = [(panels[pi].cells[int(i)],int(c)) for i,c in zip(ids,coef)]
                    self.inventory.append(dict(domain=domain,family=family,contrast=j-start+1,
                                               cells=str(cells),reference_exact=str(exact[j]),
                                               reference=values[j],true_sign=signs[j],small=bool(small[j])))
            self.parts.append((values,signs,small))

    def score(self,channels,panels,targets,ix,gp,ridge):
        scored = []
        for ch,t,(truth,signs,small),gp_pred in zip(channels,targets,self.parts,gp):
            local_paid = np.isin(ch.rows,ix)
            rr = np.empty(len(ch.rows))
            for pi in ch.panel_ids:
                p = panels[pi]
                use = p.eligible[ch.rows]
                rr[use] = ridge[p.name][ch.rows[use]]
            predicted = t.C@np.column_stack([gp_pred,rr])
            resolved = ~np.any(t.C[:,~local_paid]!=0,axis=1)
            predicted[resolved] = truth[resolved,None]
            wrong = np.sign(predicted)!=signs[:,None]
            errors = (predicted-truth[:,None])**2
            for _,_,_,a,b,scale in t.groups:
                eligible = signs[a:b]!=0
                low = small[a:b]
                group = wrong[a:b]
                mse = errors[a:b].mean(axis=0)
                scored.append([group[eligible].mean(axis=0),mse/scale,group.mean(axis=0),
                               group[low].mean(axis=0),mse])
        return np.transpose(scored,(2,0,1))

def run(site,start,stop,omit_plot=False,output_root=None,*,loader=load_trial,trial_seed=None):
    if omit_plot and site!='tcse':
        raise ValueError('The recorded plot-label sensitivity applies to TCSE.')
    folder = (output_root if output_root is not None else OUT)/(site+'_no_plot' if omit_plot else site)
    folder.mkdir(exist_ok=True,parents=True)
    trial_seed = SEEDS[site] if trial_seed is None else trial_seed
    frame,panels,channels,responses,fractions = loader(site)
    if omit_plot:
        for ch in channels:
            ch.kernel=discrepancy_kernel(frame.iloc[ch.rows].assign(plot='unavailable'),ch.x)
    targets = [TargetSet(panels,c) for c in channels]
    evaluator = Evaluator(panels,channels,targets,fractions)
    ratio = json.loads((ROOT/'spectral/external_parameter.json').read_text())['ratio']
    covs = [c.kernel+ratio*np.eye(len(c.x)) for c in channels]
    n = len(frame)
    budgets = [(n*i+5)//10 for i in range(1,11)]
    target_frame,equivalent,_ = metadata_labels(frame,panels)
    coords = [[name,f] for t in targets for _,name,f,*_ in t.groups]
    config = dict(site=site,records=n,coordinates=coords,budgets=budgets,designs=RULES,metrics=METRICS,
                  models=MODELS,seed=trial_seed,ratio=ratio,replicates=200,target_partition_equal=equivalent)
    if start==0:
        (folder/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
        pd.DataFrame(evaluator.inventory).to_csv(folder/'contrast_inventory.csv',index=False)
        save=frame.copy()
        for ch in channels:
            save.loc[ch.rows,'x_'+ch.name]=ch.x
            save.loc[ch.rows,'reference_'+ch.name]=[str(fractions[panels[ch.panel_ids[0]].name][i]) for i in ch.rows]
        save.to_csv(folder/'frame.csv',index=False)
    for rep in range(start,stop):
        dest = folder/f'draw_{rep:03d}.npz'
        if dest.is_file():
            continue
        pilot = np.random.default_rng(trial_seed+rep).choice(n,budgets[0],replace=False)
        paid = {k:v[pilot] for k,v in responses.items()}
        settings = {p.name:original.pilot_settings(p,pilot,paid[p.name]) for p in panels}
        scores = np.empty((len(RULES),len(budgets),len(MODELS),len(coords),len(METRICS)))
        selections = np.zeros((len(RULES),len(budgets),n),bool)
        paths = {}
        # Cache only final reconstructions for identical paid sets within this pilot.
        cache = {}
        for di,rule in enumerate(RULES):
            group_frame = target_frame if rule in ['target_cell_balance','a4_loo_targetcell'] else frame
            kind = {'target_cell_balance':'cell_balance','a4_loo_targetcell':'a4_loo_cell'}.get(rule,rule)
            seed = trial_seed+1000000*(1+RULES.index(kind))+rep
            if rule in MODEL_RULES:
                # Identical randomized tie priority across model-based objectives.
                oracle=original.PaidOracle(responses)
                path=select_path(frame,channels,targets,ratio,pilot,oracle.reveal(pilot),oracle.reveal,
                                 panels,rule,trial_seed+100000+rep)
                assert len(oracle.paid)==n
            elif not rule.startswith('a2_') and not rule.startswith('a4_'):
                oracle=original.PaidOracle(responses)
                snaps=original.path(group_frame,panels,oracle.reveal,pilot,oracle.reveal(pilot),settings,
                                    [n],kind,np.random.default_rng(seed))
                path=snaps[n]['selected']
            else:
                path=None
            if path is not None:
                assert len(set(path))==n and np.array_equal(path[:len(pilot)],pilot)
                paths[rule]=path.astype(np.int16)
            for bi,budget in enumerate(budgets):
                if path is None:
                    oracle=original.PaidOracle(responses)
                    ix,_,_=matched_adaptive(group_frame,panels,oracle.reveal,pilot,oracle.reveal(pilot),budget,
                                            kind,np.random.default_rng(seed+budget*1000))
                    assert len(oracle.paid)==budget
                else:
                    ix=path[:budget]
                assert len(set(ix))==budget and set(pilot)<=set(ix)
                selections[di,bi,ix]=True
                key=selections[di,bi].tobytes()
                if key not in cache:
                    ix=np.flatnonzero(selections[di,bi])
                    observed = {k:np.where(selections[di,bi],v,np.nan) for k,v in responses.items()}
                    ridge=original.reconstruct(panels,settings,ix,observed)
                    gp=[]
                    for ch,cov in zip(channels,covs):
                        iy=np.flatnonzero(selections[di,bi,ch.rows])
                        local_y=observed[panels[ch.panel_ids[0]].name][ch.rows[iy]]
                        pred=ch.x+conditional(cov,iy,local_y-ch.x[iy])
                        pred[iy]=local_y
                        gp.append(pred)
                    cache[key]=evaluator.score(channels,panels,targets,ix,gp,ridge)
                scores[di,bi]=cache[key]
        assert np.isfinite(scores).all()
        assert np.max(np.abs(scores[:,-1]))==0
        np.testing.assert_array_equal(scores[:,:,0][:,0],np.broadcast_to(scores[0,0,0],scores[:,0,0].shape))
        tmp=dest.with_suffix('.tmp')
        with tmp.open('wb') as stream:
            np.savez_compressed(stream,scores=scores,selections=np.packbits(selections,axis=-1),
                                pilot=pilot,**{'path_'+k:v for k,v in paths.items()})
        tmp.replace(dest)
        if rep%10==0: print(site,rep,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('site',choices=SEEDS)
    ap.add_argument('--start',default=0,type=int)
    ap.add_argument('--stop',default=200,type=int)
    ap.add_argument('--omit-plot',action='store_true')
    ap.add_argument('--output-dir',type=Path)
    args=ap.parse_args()
    run(args.site,args.start,args.stop,args.omit_plot,args.output_dir)
