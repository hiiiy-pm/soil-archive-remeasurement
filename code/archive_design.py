"""Finite-archive contrast, stratification and probability-sampling helpers.

Data are loaded from ../data relative to this module. Importing this module
does not create files or execute experiments. x and y keep their stored units;
in particular, MIR SOC files remain in mass percent at this loading boundary.
"""
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
FAMILIES = ['treatment', 'time', 'interaction']
DESIGNS = ['srs', 'year', 'treatment', 'year_treatment', 'x_quintile', 'year_x_tertile']
DATA_FILES = {
    'muencheberg': 'Muencheberg_SOC.csv',
    'storage': 'Storage_indicators.csv',
    'rothamsted': 'Rothamsted_cations.csv',
    'woburn': 'Woburn_cations.csv',
    'beltsville': 'Beltsville_SOC.csv',
    'lincoln': 'Lincoln_SOC.csv',
}
ALIASES = {
    'weoc': 'storage', 'wen': 'storage', 'poxc': 'storage',
    'rlt_joint': 'rothamsted', 'wlt_joint': 'woburn', 'rlt': 'rothamsted', 'wlt': 'woburn',
    'lincoln_reap': 'lincoln',
}

def load_records(name, data_dir=None):
    """Read complete stored records without filtering or converting units."""
    key = ALIASES.get(str(name).lower(), str(name).lower())
    if key not in DATA_FILES:
        raise ValueError(f'Unknown archive: {name}')
    folder = DATA_DIR if data_dir is None else Path(data_dir)
    # None is a literal Lincoln treatment label; only empty cells mean missing.
    return pd.read_csv(folder / DATA_FILES[key], keep_default_na=False, na_values=[''])

def datasets(data_dir=None):
    """Yield the four single-indicator archives in their analysis row order.

    Muencheberg rows require finite historical/reference pairs. Joint cation
    frames and the depth-specific MIR tests are obtained through load_records.
    """
    d = load_records('muencheberg', data_dir)
    d = d[np.isfinite(d.x) & np.isfinite(d.y)].sort_values(['year','treatment','plot']).reset_index(drop=True)
    yield 'muencheberg', d, 'g/kg'
    a = load_records('storage', data_dir)
    for label, xc, yc in [('weoc','weoc_2022','weoc_2025'),('wen','wen_2022','wen_2025'),('poxc','poxc_before','poxc_2025')]:
        z = pd.DataFrame(dict(year=a.year, treatment=a.cover_crop, block=a.block,
            plot=a.sample_id.str.replace(r'_\d{4}$','',regex=True), x=a[xc], y=a[yc]))
        yield label, z.sort_values(['year','treatment','plot']).reset_index(drop=True), 'mg/kg'

def framework(d):
    cells=sorted(set(zip(d.year,d.treatment)))
    lookup={v:i for i,v in enumerate(cells)}
    g=np.array([lookup[v] for v in zip(d.year,d.treatment)])
    n=np.bincount(g,minlength=len(cells))
    defs=[]
    years=sorted(set(d.year)); treatments=sorted(set(d.treatment))
    for t in years:
        for a,b in combinations(treatments,2):
            if (t,a) in lookup and (t,b) in lookup:
                defs.append(('treatment',t,t,str(a),str(b),[lookup[t,a],lookup[t,b]],[1.,-1.]))
    for a in treatments:
        for t,u in combinations(years,2):
            if (t,a) in lookup and (u,a) in lookup:
                defs.append(('time',t,u,str(a),str(a),[lookup[u,a],lookup[t,a]],[1.,-1.]))
    for t,u in combinations(years,2):
        for a,b in combinations(treatments,2):
            if all(v in lookup for v in [(t,a),(t,b),(u,a),(u,b)]):
                defs.append(('interaction',t,u,str(a),str(b),[lookup[u,a],lookup[u,b],lookup[t,a],lookup[t,b]],[1.,-1.,-1.,1.]))
    K={}
    for fam in FAMILIES:
        z=[v for v in defs if v[0]==fam]
        K[fam]=(np.array([v[5] for v in z],dtype=int),np.array([v[6] for v in z],dtype=float))
    return cells,g,n,K,defs

def strata(d,design):
    if design=='srs': keys=np.zeros(len(d),int)
    elif design=='year':keys=d.year
    elif design=='treatment':keys=d.treatment
    elif design=='year_treatment':keys=list(zip(d.year,d.treatment))
    elif design=='x_quintile':keys=pd.qcut(d.x,5,labels=False,duplicates='drop').fillna(0)
    elif design=='year_x_tertile':
        q=d.groupby('year').x.transform(lambda x:pd.qcut(x,3,labels=False,duplicates='drop').fillna(0))
        keys=list(zip(d.year,q))
    else:raise ValueError(design)
    codes=pd.factorize(pd.Series(list(keys),dtype=object),sort=False)[0]
    return [np.flatnonzero(codes==v) for v in sorted(set(codes))]

def sample_design(groups,budget,rng):
    nh=np.array([len(h) for h in groups]);q=0
    while np.minimum(nh,q+1).sum()<=budget and q<nh.max():q+=1
    base=np.minimum(nh,q);eligible=np.flatnonzero(nh>q);r=budget-int(base.sum())
    take=base.copy(); expected=base.astype(float)
    if len(eligible):
        expected[eligible]+=r/len(eligible)
        if r:take[rng.choice(eligible,r,replace=False)]+=1
    ix=[];pi=[]
    for group,k,e in zip(groups,take,expected):
        if k:
            picked=rng.choice(group,int(k),replace=False);ix.extend(picked);pi.extend([e/len(group)]*int(k))
    ix=np.array(ix,int);pi=np.array(pi,float)
    assert len(ix)==budget and len(set(ix))==budget and np.all(pi>0)
    return ix,pi

def features(d):
    x=d.x.to_numpy(float); sx=np.std(x); xs=(x-np.mean(x))/(sx or 1.)
    yf=pd.get_dummies(d.year,dtype=float).to_numpy()
    return {'affine':np.column_stack([np.ones(len(d)),xs]),'year_affine':np.column_stack([yf,xs])}

def aggregate(values,g,n):return np.bincount(g,weights=values,minlength=len(n))/n

def contrasts(mu,K):
    return {k:(mu[...,i]*w).sum(axis=-1) for k,(i,w) in K.items()}

def score(pred,truth):
    err=pred-truth
    nonzero=np.abs(truth)>1e-12 # arithmetic zero, not a substantive effect threshold
    wrong=(pred[...,nonzero]*truth[nonzero]<0)|(np.abs(pred[...,nonzero])<=1e-12)
    scale=np.mean(truth**2)
    return dict(rmse=np.sqrt(np.mean(err**2,axis=-1)),mse=np.mean(err**2,axis=-1),normalized_mse=np.mean(err**2,axis=-1)/scale if scale else np.full(pred.shape[0],np.nan),direction_disagreement=wrong.mean(axis=-1),weighted_direction_disagreement=(wrong*np.abs(truth[nonzero])).sum(axis=-1)/np.abs(truth[nonzero]).sum())
