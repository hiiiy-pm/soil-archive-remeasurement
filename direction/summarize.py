"""Summarize entire draw curves with the common-pilot pairing preserved."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent

def calculate(site):
    folder = HERE/'results'/site
    config = json.loads((folder/'config.json').read_text())
    paths = sorted(folder.glob('draw_*.npz'))
    assert len(paths)==config['replicates']==200
    cube = []
    for path in paths:
        with np.load(path) as saved:
            cube.append(saved['scores'])
            sets=np.unpackbits(saved['selections'],axis=-1,count=config['records'])
            assert np.all(sets.sum(axis=-1)==np.array(config['budgets']))
            assert sets[:,:,saved['pilot']].all()
    raw = np.stack(cube)
    coords = config['coordinates']
    domains = sorted({d for d,f in coords})
    design = config['designs']
    wide = []
    for j,(domain,family) in enumerate(coords):
        mean = raw[:,:,:-1,:,j,:].mean(axis=(0,2))
        for di,name in enumerate(design):
            for mi,model in enumerate(config['models']):
                wide.append(dict(domain=domain,family=family,design=name,model=model,
                                 **dict(zip(config['metrics'],mean[di,mi]))))
    pd.DataFrame(wide).to_csv(folder/'domain_means.csv',index=False)
    summaries = []
    for group in ['primary','treatment','time','interaction']:
        weights = np.zeros(len(coords))
        for domain in domains:
            indices=[j for j,(d,f) in enumerate(coords) if d==domain and
                     (f in ['treatment','time'] if group=='primary' else f==group)]
            if indices:
                weights[indices]=1/len(indices)
        weights /= weights.sum()
        joint = np.einsum('rdbmjk,j->rdbmk',raw,weights)
        periods = [(str(b),[j]) for j,b in enumerate(config['budgets'])]
        periods.extend([('noncensus',slice(None,-1)),('after_pilot',slice(1,-1))])
        for period,ix in periods:
            values=joint[:,:,ix,:,:].mean(axis=2)
            focal=values[:,design.index('sign_error'),0]
            delta=focal[:,None,None,:]-values
            avg=values.mean(axis=0)
            diff=delta.mean(axis=0)
            se=delta.std(axis=0,ddof=1)/np.sqrt(len(delta))
            z=norm.ppf(1-.05/(2*2*(len(design)-1)))
            for di,name in enumerate(design):
                for mi,model in enumerate(config['models']):
                    for k,metric in enumerate(config['metrics'][:-1]):
                        d=diff[di,mi,k]
                        e=se[di,mi,k]
                        summaries.append(dict(site=site,group=group,budget=period,model=model,design=name,
                            metric=metric,mean=avg[di,mi,k],sign_minus_comparator=d,mcse=e,
                            lower=d-1.96*e,upper=d+1.96*e,
                            simultaneous_lower=d-z*e,simultaneous_upper=d+z*e))
    result=pd.DataFrame(summaries)
    result.to_csv(folder/'paired_summary.csv',index=False)
    main=result[(result.group=='primary')&(result.budget=='noncensus')&
                result.metric.isin(['direction','scaled_mse'])]
    print(site,main[main.model.eq('kriging')].pivot(index='design',columns='metric',values='mean').round(6).to_string())
    return result

if __name__=='__main__':
    import sys
    for site in sys.argv[1:] or ['tcse','ultuna','tcse_no_plot']:
        calculate(site)
