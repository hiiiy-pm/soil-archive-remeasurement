"""Paired reconstruction effects and finite-archive error decomposition.

Uses saved designs; full reference values are for retrospective scoring only.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pilot_risk as pr
from run_experiment import UNIVERSES,outputs,score,METHODS,METRICS
H=Path(__file__).resolve().parents[1]
def main():
    dest=H/'diagnostics';dest.mkdir(exist_ok=True)
    summaries=[];decomp=[];reg=[]
    for name in UNIVERSES:
        d,panels,y=pr.load(name);c=json.loads((H/'results'/name/'config.json').read_text());N=len(d)
        truth={p.name:pr.engine.design.contrasts(pr.engine.aggregate_panel(p,y[p.name]),p.K) for p in panels}
        truthmu={p.name:pr.engine.aggregate_panel(p,y[p.name]) for p in panels}
        arrays=[];cellerrors={}
        for rep in range(200):
            saved=np.load(H/'results'/name/f'draw_{rep:04}.npz');log=json.loads((H/'results'/name/f'draw_{rep:04}.json').read_text())
            pure=np.empty_like(saved['scores']);retained=saved['scores']
            for di,design in enumerate(c['designs']):
                for bi,B in enumerate(c['budgets']):
                    ix=saved[f'path__{design}'][:B] if f'path__{design}' in saved else saved[f'indices__{design}__{B}']
                    ob={p.name:np.full(N,np.nan) for p in panels}
                    pred={m:{} for m in METHODS}
                    for p in panels:
                        ii=ix[p.eligible[ix]];ob[p.name][ix]=y[p.name][ix];z=p.F['year_affine']
                        beta=np.linalg.solve(log['settings'][p.name]['lam']*np.eye(z.shape[1])+z[ii].T@z[ii],z[ii].T@(y[p.name][ii]-p.x[ii]))
                        f=p.x+z@beta;pred['pilot_ridge'][p.name]=f
                        pred['year_affine'][p.name]=pr.engine.fit_predict(p,ix,ob[p.name],'year_affine')[0]
                        if B==c['budgets'][4]:
                            anchored=f.copy();anchored[ii]=y[p.name][ii]
                            cellerrors.setdefault((design,p.name),[]).append(pr.engine.aggregate_panel(p,anchored)-truthmu[p.name])
                    pure[di,bi]=score(panels,truth,pred,c['coordinates'])
            arrays.append(pure)
        a=np.stack(arrays)
        retained=np.stack([np.load(H/'results'/name/f'draw_{rep:04}.npz')['scores'] for rep in range(200)])
        names=list(dict.fromkeys(p for p,f in c['coordinates']))
        def joint(v):return np.mean([np.mean(v[:,:,:,:,[i for i,(p,f) in enumerate(c['coordinates']) if p==n],:],axis=4) for n in names],axis=0)
        jr=joint(retained);jp=joint(a)
        for di,design in enumerate(c['designs']):
            for mi,m in enumerate(METHODS):
                for ki,k in enumerate(METRICS):
                    if k=='mse':continue
                    delta=(jr[:,di,:-1,mi,ki]-jp[:,di,:-1,mi,ki]).mean(axis=1)
                    summaries.append(dict(universe=name,design=design,method=m,metric=k,retained=jr[:,di,:-1,mi,ki].mean(),pure=jp[:,di,:-1,mi,ki].mean(),difference=delta.mean(),mcse=delta.std(ddof=1)/np.sqrt(200)))
            for ki,k in enumerate(METRICS):
                if k=='mse':continue
                delta=(jr[:,di,:-1,0,ki]-jr[:,di,:-1,1,ki]).mean(axis=1)
                reg.append(dict(universe=name,design=design,metric=k,ridge=jr[:,di,:-1,0,ki].mean(),ols=jr[:,di,:-1,1,ki].mean(),difference=delta.mean(),mcse=delta.std(ddof=1)/np.sqrt(200)))
            bias=0.;variance=0.
            for p in panels:
                e=np.stack(cellerrors[(design,p.name)]);mu=e.mean(axis=0)
                bias+=mu@p.G@mu;variance+=np.mean(np.einsum('ij,ij->i',(e-mu)@p.G,e-mu))
            mean=jr[:,di,4,0,4].mean()
            assert np.isclose(mean,bias+variance,rtol=1e-8,atol=1e-10)
            decomp.append(dict(universe=name,budget=c['budgets'][4],design=design,history_scaled_mse=mean,squared_mean_error=bias,allocation_variance=variance))
        print('diagnostics',name,flush=True)
    for fn,rows in [('retention_effects',summaries),('regularization_effects',reg),('error_decomposition',decomp)]:pd.DataFrame(rows).to_csv(dest/(fn+'.csv'),index=False)
if __name__=='__main__':main()
