"""Checkpointed paid-pilot experiment; all later y is available only to scoring."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import argparse,json,time,hashlib,sys
import numpy as np
import pandas as pd
import pilot_risk as pr
from comparators import matched_adaptive,DESIGNS as ADAPTIVE
from target_cells import metadata_labels

H=Path(__file__).resolve().parents[1]
UNIVERSES=['muencheberg','weoc','wen','poxc','rothamsted','woburn','beltsville','lincoln']
DESIGNS=pr.PATH_DESIGNS+list(ADAPTIVE)+['target_cell_balance','a4_loo_targetcell']
SEED_BASE=202611160000  # Fixed integer namespace; it is not a calendar input.
METHODS=['pilot_ridge','year_affine']
METRICS=['mse','direction_disagreement','weighted_direction_disagreement','normalized_mse','historical_scaled_mse']

def file_sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def outputs(panels,selected,predictions,observed):
    out={'pilot_ridge':predictions,'year_affine':{}}
    for p in panels:
        f,*_=pr.engine.fit_predict(p,selected,observed[p.name],'year_affine')
        ii=selected[p.eligible[selected]];f[ii]=observed[p.name][ii]
        out['year_affine'][p.name]=f
    return out

def score(panels,truth,out,coordinates):
    result=np.empty((len(METHODS),len(coordinates),len(METRICS)))
    for mi,method in enumerate(METHODS):
        scored={}
        for p in panels:
            mu=pr.engine.aggregate_panel(p,out[method][p.name])
            scored[p.name]=pr.engine.score(p,mu,truth[p.name])
            for f,v in scored[p.name].items():v['historical_scaled_mse']=v['mse']/p.scale
        for ci,(name,fam) in enumerate(coordinates):
            result[mi,ci]=[scored[name][fam][metric] for metric in METRICS]
    return result

def execute(name,reps,resume=True,output_root=None):
    folder=(H/'results' if output_root is None else Path(output_root))/name;folder.mkdir(parents=True,exist_ok=True)
    d,panels,truth_values=pr.load(name);N=len(d)
    target_d,partition_equivalent,target_signatures=metadata_labels(d,panels)
    budgets=sorted(set(int(np.floor(N*f+.5)) for f in np.arange(.1,1.01,.1)))
    pilot_n=budgets[0];coords=[(p.name,f) for p in panels for f in p.K]
    truth={p.name:pr.engine.design.contrasts(pr.engine.aggregate_panel(p,truth_values[p.name]),p.K) for p in panels}
    # The archived fingerprints use CRLF, including when replayed on other platforms.
    fingerprint=hashlib.sha256(d.to_csv(index=False, lineterminator='\r\n').encode())
    for p in panels:
        for z in [p.x,p.eligible,truth_values[p.name]]:fingerprint.update(z.tobytes())
    config=dict(universe=name,N=N,budgets=budgets,pilot_n=pilot_n,coordinates=coords,
                designs=DESIGNS,methods=METHODS,metrics=METRICS,data_fingerprint=fingerprint.hexdigest(),
                seed_base=SEED_BASE,seed_namespace='fixed_integer',universe_seed_offset=UNIVERSES.index(name)*1000000,
                seed_rule=dict(pilot='base + repeat',path='base + 100000000 + seed_design_index*10000 + repeat',adapter='base + 200000000 + seed_design_index*1000000 + budget*1000 + repeat'),
                seed_design_index={n:DESIGNS.index({'target_cell_balance':'cell_balance','a4_loo_targetcell':'a4_loo_cell'}.get(n,n)) for n in DESIGNS},
                target_partition_equivalent=partition_equivalent,target_cell_count=len(set(target_signatures)))
    conf=folder/'config.json'
    if conf.exists():
        old=json.loads(conf.read_text());assert old==json.loads(json.dumps(config)), 'Configuration changed; use a separate output directory'
    else:conf.write_text(json.dumps(config,indent=2),encoding='utf-8')
    base=SEED_BASE+UNIVERSES.index(name)*1000000
    started=time.time()
    for rep in range(reps):
        target=folder/f'draw_{rep:04}.npz';logpath=folder/f'draw_{rep:04}.json'
        if resume and target.exists() and logpath.exists():continue
        pilot=np.random.default_rng(base+rep).choice(N,pilot_n,replace=False)
        first=pr.PaidOracle(truth_values);paid=first.reveal(pilot)
        settings={p.name:pr.pilot_settings(p,pilot,paid[p.name]) for p in panels}
        arrays={};scores=np.empty((len(DESIGNS),len(budgets),len(METHODS),len(coords),len(METRICS)))
        risks=np.full((len(DESIGNS),len(budgets)),np.nan);waves={};duration={}
        for di,design in enumerate(DESIGNS):
            start=time.time()
            actual_design={'target_cell_balance':'cell_balance','a4_loo_targetcell':'a4_loo_cell'}.get(design,design)
            seed_di=DESIGNS.index(actual_design)
            allocation_d=target_d if design in ['target_cell_balance','a4_loo_targetcell'] else d
            if actual_design in pr.PATH_DESIGNS:
                oracle=pr.PaidOracle(truth_values);init=oracle.reveal(pilot)
                snaps=pr.path(allocation_d,panels,oracle.reveal,pilot,init,settings,budgets,actual_design,np.random.default_rng(base+100000000+seed_di*10000+rep))
                assert len(oracle.paid)==N
                arrays[f'path__{design}']=snaps[N]['selected']
                for bi,B in enumerate(budgets):
                    snap=snaps[B];ix=snap['selected'];assert len(ix)==B and len(set(ix))==B and set(pilot).issubset(ix)
                    ob={p.name:np.full(N,np.nan) for p in panels}
                    # Evaluation constructs a masked observation view from exactly
                    # the paid indices. It is never sent back into the selector.
                    for p in panels:ob[p.name][ix]=truth_values[p.name][ix]
                    out=outputs(panels,ix,snap['predictions'],ob)
                    scores[di,bi]=score(panels,truth,out,coords);risks[di,bi]=snap['predicted_risk']
            else:
                for bi,B in enumerate(budgets):
                    oracle=pr.PaidOracle(truth_values);init=oracle.reveal(pilot)
                    ix,ob,logs=matched_adaptive(allocation_d,panels,oracle.reveal,pilot,init,B,actual_design,np.random.default_rng(base+200000000+seed_di*1000000+B*1000+rep))
                    assert len(oracle.paid)==B and len(ix)==B and set(pilot).issubset(ix)
                    predictions=pr.reconstruct(panels,settings,ix,ob)
                    out=outputs(panels,ix,predictions,ob)
                    scores[di,bi]=score(panels,truth,out,coords)
                    arrays[f'indices__{design}__{B}']=ix
                    waves[f'{design}__{B}']=logs
            duration[design]=time.time()-start
        assert np.allclose(scores[:,0],scores[0,0],rtol=1e-8,atol=1e-9,equal_nan=True), 'Common pilot comparison not identical'
        assert np.max(np.abs(scores[:,-1,:,:,0]))<1e-20, 'Census failed'
        arrays.update(scores=scores,predicted_risk=risks,pilot=pilot)
        tmp=target.with_suffix('.tmp')
        with tmp.open('wb') as stream:np.savez_compressed(stream,**arrays)
        os.replace(tmp,target)
        logged_settings = {
            panel: {key: value if np.isfinite(value) else None for key, value in values.items()}
            for panel, values in settings.items()
        }
        passed, r2 = pr.gate(settings)
        log=dict(rep=rep,pilot_seed=base+rep,settings=logged_settings,
                 active_gate=[passed, r2 if np.isfinite(r2) else None],
                 seconds=duration,waves=waves,results_sha256=file_sha(target))
        logpath.write_text(json.dumps(log,indent=2,allow_nan=False),encoding='utf-8')
        if rep%10==0 or rep==reps-1:print(name,rep+1,'/',reps,'elapsed',round(time.time()-started,1),'s',flush=True)
    return name

if __name__=='__main__':
    ap=argparse.ArgumentParser(description='Run the 12-arm, two-reconstruction paid-pilot experiment.')
    ap.add_argument('--universe',choices=UNIVERSES+['all'],required=True)
    ap.add_argument('--reps',type=int,default=200)
    ap.add_argument('--output-dir',type=Path)
    ap.add_argument('--no-resume',action='store_true')
    a=ap.parse_args()
    if a.reps<1:ap.error('--reps must be positive')
    for name in UNIVERSES if a.universe=='all' else [a.universe]:
        execute(name,a.reps,not a.no_resume,a.output_dir)
