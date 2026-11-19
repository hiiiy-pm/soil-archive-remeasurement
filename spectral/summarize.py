from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent

def estimate(values):
    avg=values.mean()
    se=values.std(ddof=1)/np.sqrt(len(values))
    return dict(mean=avg,mcse=se,lower=avg-1.96*se,upper=avg+1.96*se)

curves=[]
budgets=[]
families=[]
differences=[]
for site in ['beltsville','lincoln']:
    with np.load(HERE/(site+'.npz')) as data:
        raw=data['scores']
        joint=np.einsum('rdpmfk,f->rdpmk',raw,data['joint_weights'])
        names=list(data['designs'])
        new=names.index('pooled_retained_contrast')
        scores=joint[:,:,:-1].mean(axis=2)
        for di,name in enumerate(names):
            for mi,model in enumerate(data['reconstructions']):
                for ki,metric in enumerate(data['metrics']):
                    ids=dict(site=site,selection=name,reconstruction=model,metric=metric)
                    curves.append({**ids,**estimate(scores[:,di,mi,ki])})
                    for bi,b in enumerate(data['budgets']):
                        budgets.append({**ids,'budget':b,**estimate(joint[:,di,bi,mi,ki])})
                    for fi,(domain,fam) in enumerate(data['coordinates']):
                        families.append({**ids,'domain':domain,'family':fam,
                                         **estimate(raw[:,di,:-1,mi,fi,ki].mean(axis=1))})
                    differences.append({**ids,**estimate(scores[:,new,1,ki]-scores[:,di,mi,ki])})
for rows,fn in [(curves,'curve_scores'),(budgets,'budget_scores'),(families,'domain_family_scores'),(differences,'paired_differences')]:
    pd.DataFrame(rows).to_csv(HERE/(fn+'.csv'),index=False)
z=pd.DataFrame(curves)
for site in ['beltsville','lincoln']:
    s=z[z.site==site]
    print(site,s.pivot(index='selection',columns=['reconstruction','metric'],values='mean').round(5).to_string())
