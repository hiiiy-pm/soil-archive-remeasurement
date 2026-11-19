"""Finite soil-archive panels, fitted reconstruction, and contrast scoring."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
import archive_design as design
from integer_allocation import allocate_a_optimal

FAMILIES=['treatment','time','interaction']

@dataclass
class Panel:
    name: str
    unit: str
    eligible: np.ndarray
    rows: np.ndarray
    x: np.ndarray
    g: np.ndarray
    n: np.ndarray
    cells: list
    K: dict
    F: dict
    xmean: np.ndarray
    G: np.ndarray
    scale: float
    scale_rule: str

def make_panel(name,d,x,eligible,unit,panel_count):
    rows=np.flatnonzero(eligible)
    z=d.iloc[rows].copy().reset_index(drop=True);z['x']=x[rows]
    cells,g,n,K,defs=design.framework(z)
    K={f:v for f,v in K.items() if len(v[0])}
    nf=len(K);G=np.zeros((len(n),len(n)))
    for f,(idx,w) in K.items():
        for a in range(idx.shape[1]):
            for b in range(idx.shape[1]):np.add.at(G,(idx[:,a],idx[:,b]),w[:,a]*w[:,b]/(nf*len(idx)))
    scale=float(np.var(x[rows],ddof=0));rule='historical_variance_ddof0'
    if scale==0:scale=float(np.mean(x[rows]**2));rule='historical_mean_square'
    if scale==0:scale=1.;rule='unit_fallback_all_historical_zero'
    G/=panel_count*scale
    fullg=np.full(len(d),-1,int);fullg[rows]=g
    Fsmall=design.features(z)
    F={k:np.zeros((len(d),v.shape[1])) for k,v in Fsmall.items()}
    for k,v in Fsmall.items():F[k][rows]=v
    F['additive']=np.zeros((len(d),1));F['additive'][rows]=1.
    F['year_additive']=np.zeros((len(d),Fsmall['year_affine'].shape[1]-1));F['year_additive'][rows]=Fsmall['year_affine'][:,:-1]
    xx=np.array(x,float);xx[~eligible]=0.
    return Panel(name,unit,np.array(eligible),rows,xx,fullg,n,cells,K,F,design.aggregate(xx[rows],g,n),G,scale,rule)

def aggregate_panel(p,values):return np.bincount(p.g[p.rows],weights=values[p.rows],minlength=len(p.n))/p.n

def fit_predict(p,selected,obs,model,weights=None,loo=False):
    ix=np.asarray(selected,int);ix=ix[p.eligible[ix]]
    f=p.x.copy()
    if model=='observed_only':return f,0,np.empty(0),ix,0,0
    if model=='year_ratio':
        miss=0
        for j in range(p.F['year_additive'].shape[1]):
            allrows=np.flatnonzero(p.F['year_additive'][:,j]);ii=ix[p.F['year_additive'][ix,j]>0]
            den=p.x[ii].sum()
            if len(ii)==0 or den==0:miss+=1
            else:f[allrows]=p.x[allrows]*(obs[ii].sum()/den)
        return f,int(miss>0),np.empty(0),ix,miss,0
    Z=p.F[model];zs=Z[ix];isadd='additive' in model
    target=obs[ix]-p.x[ix] if isadd else obs[ix]
    rank=np.linalg.matrix_rank(zs) if len(ix) else 0
    if rank<Z.shape[1]:return f,1,obs[ix]-p.x[ix],ix,len(ix) if loo else 0,0
    if weights is None:zw=zs;yw=target
    else:
        w=np.sqrt(weights[ix]);zw=zs*w[:,None];yw=target*w
    coef=np.linalg.lstsq(zw,yw,rcond=None)[0]
    f=Z@coef+(p.x if isadd else 0.)
    residual=obs[ix]-f[ix];loo_fallback=0;loo_refits=0
    if loo:
        # Orthogonal projection gives leverage without squaring condition number.
        u=np.linalg.svd(zw,full_matrices=False)[0];lev=(u[:,:rank]**2).sum(axis=1)
        regular=1-lev>1e-10
        rr=obs[ix]-p.x[ix];rr[regular]=residual[regular]/(1-lev[regular])
        for j in np.flatnonzero(~regular):
            keep=np.arange(len(ix))!=j;zminus=zs[keep];loo_refits+=1
            minus_rank=np.linalg.matrix_rank(zminus) if len(zminus) else 0
            if minus_rank<Z.shape[1]:
                loo_fallback+=1
            else:
                cm=np.linalg.lstsq(zw[keep],yw[keep],rcond=None)[0]
                pred=zs[j]@cm+(p.x[ix[j]] if isadd else 0.)
                rr[j]=obs[ix[j]]-pred
        residual=rr
    return f,0,residual,ix,loo_fallback,loo_refits

def selection_features(d,panels):
    columns=[]
    for p in panels:
        v=np.zeros(len(d));v[p.eligible]=(p.x[p.eligible]-p.x[p.eligible].mean())/np.sqrt(p.scale)
        columns.append(v)
        if not p.eligible.all():columns.append((~p.eligible).astype(float))
    X=np.column_stack(columns)
    if X.shape[1]==1:pc=X[:,0]
    else:
        u,s,v=np.linalg.svd(X-X.mean(axis=0),full_matrices=False);pc=u[:,0]*s[0]
        if v[0,np.argmax(np.abs(v[0]))]<0:pc=-pc
    frame=d[['year','treatment']].copy();frame['x']=pc
    return X,frame

def balanced_counts(N,already,budget,rng):
    n=already.copy()
    for _ in range(int(budget-n.sum())):
        eligible=np.flatnonzero(n<N);small=eligible[n[eligible]==n[eligible].min()]
        n[int(rng.choice(small))]+=1
    return n

def influence_variance(panels,values,selected,groups):
    """Weighted trace of covariance of complete target contributions C_i v_i."""
    var=np.full(len(groups),np.nan)
    selectedmask=np.zeros(len(panels[0].x),bool);selectedmask[selected]=True
    for h,group in enumerate(groups):
        ii=group[selectedmask[group]];m=len(ii)
        if m<2:continue
        total=0.
        for p in panels:
            jj=ii[p.eligible[ii]]
            if not len(jj):continue
            v=values[p.name][jj]/p.n[p.g[jj]]
            b=np.bincount(p.g[jj],weights=v,minlength=len(p.n))
            sumsq=np.dot(np.diag(p.G)[p.g[jj]],v*v)
            # Only occupied cells are needed, avoiding large sparse matrix products.
            nz=np.flatnonzero(b);norm=b[nz]@p.G[np.ix_(nz,nz)]@b[nz]
            total+=(sumsq-norm/m)/(m-1)
        var[h]=max(0.,float(total)) # nonnegative variance up to floating roundoff
    return var

def score(p,mu,truth):
    values=design.contrasts(mu,p.K);out={}
    for family,v in values.items():
        t=truth[family];err=v-t;valid=np.abs(t)>1e-12
        bad=(v[valid]*t[valid]<0)|(np.abs(v[valid])<=1e-12)
        out[family]=dict(mse=float(np.mean(err**2)),direction_disagreement=float(bad.mean()) if valid.any() else np.nan,
            weighted_direction_disagreement=float(np.dot(bad,np.abs(t[valid]))/np.abs(t[valid]).sum()) if valid.any() else np.nan,
            normalized_mse=float(np.mean(err**2)/np.mean(t**2)) if np.mean(t**2)>0 else np.nan)
    return out

def load_universe(name):
    """Build known-data panels and a separate paid-reference response dictionary."""
    aliases={'rlt_joint':'rothamsted','wlt_joint':'woburn','lincoln_reap':'lincoln'}
    name=aliases.get(name,name)
    if name in ['muencheberg','weoc','wen','poxc']:
        _,d,unit=next(t for t in design.datasets() if t[0]==name)
        d=d.copy();x=d.x.to_numpy();y=d.y.to_numpy();d=d.drop(columns=['x','y'])
        p=make_panel(name,d,x,np.ones(len(d),bool),unit,1)
        return d,[p],{name:y}
    d=design.load_records(name)
    if name in ['beltsville','lincoln']:
        assert set(d.units)=={'SOC_percent_mass'}
        # Stored source percentages are converted exactly once at this boundary.
        response=d.y.to_numpy(float)*10;x=d.x.to_numpy(float)*10
        public=d.drop(columns=[c for c in d if c in ['x','y'] or c.startswith('y_')])
        panels=[];responses={}
        domains=sorted(d.depth.unique()) if name=='lincoln' else [None]
        for domain in domains:
            eligible=np.ones(len(d),bool) if domain is None else (d.depth==domain).to_numpy()
            label=name if domain is None else name+'_depth_'+str(domain).replace(' ','').replace('.','p')
            panels.append(make_panel(label,public,x,eligible,'g/kg',len(domains)))
            responses[label]=response.copy()
        return public,panels,responses
    if name not in ['rothamsted','woburn']:
        raise ValueError(name)
    panels=[];responses={}
    for indicator in ['Ca','K','Mg','Na','Mn']:
        eligible=d[f'pair_eligible_{indicator}'].to_numpy(bool)
        x=d[f'x_{indicator}'].to_numpy(float);y=d[f'y_{indicator}'].to_numpy(float)
        assert np.all(np.isfinite(x[eligible])) and np.all(np.isfinite(y[eligible]))
        key=name+'_'+indicator.lower()
        panels.append(make_panel(key,d,x,eligible,'mg/kg dry soil',5))
        responses[key]=y
    public=d.drop(columns=[c for c in d if c=='y' or c.startswith('y_')])
    assert not any(c=='y' or c.startswith('y_') for c in public)
    return public,panels,responses
