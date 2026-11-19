"""Summarize paired comparisons across the full budgets and after the common pilot."""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
rows = []
family_rows = []
for site in ['beltsville','lincoln','pendleton']:
    with np.load(HERE/f'external_{site}.npz') as data:
        joint = np.einsum('rdpmfk,f->rdpmk', data['scores'],data['joint_weights'])
        names = list(data['designs'])
        candidate = joint[:,names.index('external_retained_contrast'),:,1]
        metrics = list(data['metrics'])
        reconstructions = list(data['reconstructions'])
        budgets = list(data['budgets'])
        for di,name in enumerate(names):
            for mi,model in enumerate(reconstructions):
                reference = joint[:,di,:,mi]
                periods = [(str(b), [i]) for i,b in enumerate(budgets)]
                periods.extend([('noncensus',slice(None,-1)),('after_pilot',slice(1,-1))])
                for period,indices in periods:
                    main = candidate[:,indices].mean(axis=1)
                    comparison = reference[:,indices].mean(axis=1)
                    for k,metric in enumerate(metrics):
                        diff = main[:,k]-comparison[:,k]
                        hw = 1.96*diff.std(ddof=1)/np.sqrt(len(diff))
                        rows.append(dict(site=site,selection=name,reconstruction=model,budget=period,
                                         metric=metric,mean=comparison[:,k].mean(),candidate=main[:,k].mean(),
                                         paired_difference=diff.mean(),lower=diff.mean()-hw,upper=diff.mean()+hw))
                family_scores = data['scores'][:,di,:-1,mi].mean(axis=(0,1))
                for (domain,family),values in zip(data['coordinates'],family_scores):
                    family_rows.append(dict(site=site,selection=name,reconstruction=model,
                                            domain=domain,family=family,scaled_mse=values[0],direction=values[1]))
pd.DataFrame(rows).to_csv(HERE/'external_scores.csv', index=False)
pd.DataFrame(family_rows).to_csv(HERE/'external_families.csv', index=False)
