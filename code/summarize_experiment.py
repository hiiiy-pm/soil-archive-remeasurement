"""Summaries retain paired repeat indices and equal panel/family weights."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
H=Path(__file__).resolve().parents[1]
NAMES=['muencheberg','weoc','wen','poxc','rothamsted','woburn','beltsville','lincoln']
def se(x):return np.std(x,axis=0,ddof=1)/np.sqrt(len(x))
def load_joint(name):
    folder=H/'results'/name;c=json.loads((folder/'config.json').read_text())
    files=sorted(folder.glob('draw_*.npz'));assert len(files)==200
    a=np.stack([np.load(f)['scores'] for f in files])
    names=list(dict.fromkeys(p for p,f in c['coordinates']))
    joint=np.mean([np.mean(a[:,:,:,:,[i for i,(p,f) in enumerate(c['coordinates']) if p==n],:],axis=4) for n in names],axis=0)
    return c,a,joint
def main():
    out=H/'summaries';out.mkdir(exist_ok=True)
    rows=[];curves=[];pairs=[];family=[];pilot=[]
    for name in NAMES:
        c,a,j=load_joint(name)
        for di,design in enumerate(c['designs']):
            for mi,method in enumerate(c['methods']):
                for ki,metric in enumerate(c['metrics']):
                    if metric=='mse':continue
                    q=j[:,di,:,mi,ki];cc=q[:,:-1].mean(axis=1)
                    curves.append(dict(universe=name,design=design,method=method,metric=metric,mean=cc.mean(),mcse=se(cc),repeats=len(q)))
                    for bi,B in enumerate(c['budgets']):
                        rows.append(dict(universe=name,N=c['N'],budget=B,fraction=B/c['N'],design=design,method=method,metric=metric,mean=q[:,bi].mean(),mcse=se(q[:,bi])))
                for bi,B in enumerate(c['budgets']):
                    for ci,(p,f) in enumerate(c['coordinates']):
                        for ki,metric in enumerate(c['metrics']):
                            x=a[:,di,bi,mi,ci,ki]
                            family.append(dict(universe=name,panel=p,family=f,budget=B,design=design,method=method,metric=metric,mean=x.mean(),mcse=se(x)))
                            if metric=='mse':
                                rmse=float(np.sqrt(x.mean()))
                                family.append(dict(universe=name,panel=p,family=f,budget=B,design=design,method=method,metric='pooled_rmse',mean=rmse,mcse=float(se(x)/(2*rmse)) if rmse else 0.))
        di=c['designs'].index('retained_contrast')
        for mi,method in enumerate(c['methods']):
            for bj,baseline in enumerate(c['designs']):
                for ki,metric in enumerate(c['metrics']):
                    if metric=='mse':continue
                    x=(j[:,di,:-1,mi,ki]-j[:,bj,:-1,mi,ki]).mean(axis=1)
                    pairs.append(dict(universe=name,baseline=baseline,method=method,metric=metric,mean=x.mean(),mcse=se(x),lower=x.mean()-1.96*se(x),upper=x.mean()+1.96*se(x)))
        logs=[json.loads(f.read_text()) for f in sorted((H/'results'/name).glob('draw_*.json'))]
        for rep,l in enumerate(logs):
            for panel,s in l['settings'].items():pilot.append(dict(universe=name,rep=rep,panel=panel,gate_passed=l['active_gate'][0],pilot_r2=l['active_gate'][1],**s))
    for fn,x in [('budget_scores',rows),('curve_scores',curves),('paired_curve_differences',pairs),('family_scores',family),('pilot_settings',pilot)]:pd.DataFrame(x).to_csv(out/(fn+'.csv'),index=False)
    print('Summaries written',out)
    z=pd.DataFrame(curves);z=z[(z.method=='pilot_ridge')&(z.metric=='historical_scaled_mse')]
    print(z.pivot(index='universe',columns='design',values='mean').round(5).to_string())
if __name__=='__main__':main()
