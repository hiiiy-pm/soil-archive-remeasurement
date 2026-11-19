"""Paid-pilot Gaussian reconstruction and finite-reference target acquisition.

Selection receives historical panels and a reveal callback, never a response
array. Lambda/noise are learned from the paid pilot, then frozen.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

import archive_model as engine

LAMBDAS=10.**np.arange(-4,5,dtype=float)
PATH_DESIGNS=['random','cell_balance','maximin','conversion','concentration','retained_contrast','active_sampling']

def load(name):
    return engine.load_universe(name)

def pilot_settings(p,pilot,paid):
    keep=p.eligible[pilot];ix=pilot[keep]
    z=p.F['year_affine'][ix];delta=np.asarray(paid)[keep]-p.x[ix]
    q=p.F['year_affine'].shape[1]
    if len(ix)<3:
        lam=1.;sig=p.scale;loss=np.nan;sse=np.nan;sst=np.nan;fallback=True
    else:
        losses=[];cache=[]
        for lam0 in LAMBDAS:
            v=np.linalg.inv(lam0*np.eye(q)+z.T@z)
            beta=v@z.T@delta
            h=np.einsum('ij,ij->i',z@v,z)
            one=np.maximum(1-h,np.finfo(float).eps)
            e=delta-z@beta
            losses.append(float(np.mean((e/one)**2)))
            cache.append((v,beta,h,e,one))
        best=int(np.argmin(losses));lam=float(LAMBDAS[best]);loss=losses[best]
        _,_,_,e,one=cache[best]
        sig=float(np.mean(e*e/one))
        sse=float(np.mean((e/one)**2));sst=float(np.mean((delta-delta.mean())**2));fallback=False
    sig=max(float(sig),np.finfo(float).eps*max(p.scale,1e-30))
    return dict(lam=lam,sigma2=sig,pilot_eligible=len(ix),loo_mse=loss,
                loo_mse_scaled=sse/p.scale,baseline_mse_scaled=sst/p.scale,
                sparse_pilot_fallback=fallback)

class Posterior:
    def __init__(self,p,setting,pilot,paid):
        self.p=p;self.setting=setting;self.Z=p.F['year_affine'];q=self.Z.shape[1]
        self.V=np.eye(q)/setting['lam'];self.beta=np.zeros(q)
        self.obs=np.full(len(p.x),np.nan)
        self.remaining=p.eligible.copy()
        self.B=np.column_stack([engine.aggregate_panel(p,self.Z[:,j]) for j in range(q)])
        self.GB=p.G@self.B;self.H=self.B.T@self.GB;self.Hfull=self.H.copy()
        self.E=self.Z.T@self.Z
        self.diag=np.diag(p.G)
        for i,y in zip(pilot,paid):self.update(int(i),float(y))

    def update(self,i,y):
        self.obs[i]=y
        if not self.p.eligible[i]:return
        assert self.remaining[i]
        z=self.Z[i];v=self.V@z;den=1+z@v
        self.beta+=v*((y-self.p.x[i]-z@self.beta)/den)
        self.V-=np.outer(v,v)/den
        g=self.p.g[i];a=z/self.p.n[g];u=self.GB[g].copy()
        self.H-=np.outer(a,u)+np.outer(u,a)-self.diag[g]*np.outer(a,a)
        self.GB-=self.p.G[:,g,None]*a[None,:]
        self.B[g]-=a
        self.E-=np.outer(z,z)
        self.remaining[i]=False

    def scores(self,kind):
        p=self.p;ii=np.flatnonzero(self.remaining)
        out=np.zeros(len(p.x))
        if not len(ii):return out
        z=self.Z[ii];v=z@self.V;h=np.einsum('ij,ij->i',z,v)
        if kind=='conversion':
            num=np.einsum('ij,ij->i',v@self.Hfull,v)
        elif kind=='concentration':
            num=np.einsum('ij,ij->i',v@self.E,v)+2*h+1
            num/=self.panel_count*len(p.rows)*p.scale
        elif kind=='retained_contrast':
            g=p.g[ii];n=p.n[g]
            num=np.einsum('ij,ij->i',v@self.H,v)
            num+=2*np.einsum('ij,ij->i',self.GB[g],v)/n+self.diag[g]/n**2
        else:raise KeyError(kind)
        out[ii]=self.setting['sigma2']*np.maximum(num,0)/(1+h)
        return out

    def contribution_second_moment(self):
        p=self.p;ii=np.flatnonzero(self.remaining);out=np.zeros(len(p.x))
        z=self.Z[ii];h=np.einsum('ij,ij->i',z@self.V,z)
        g=p.g[ii]
        out[ii]=self.diag[g]/p.n[g]**2*((z@self.beta)**2+self.setting['sigma2']*(1+h))
        return out

    def predictions(self):
        f=self.p.x+self.Z@self.beta
        ix=np.isfinite(self.obs)&self.p.eligible
        f[ix]=self.obs[ix]
        return f

    def risk(self):
        p=self.p;ii=np.flatnonzero(self.remaining);g=p.g[ii]
        return self.setting['sigma2']*max(0.,float(np.trace(self.V@self.H)+np.sum(self.diag[g]/p.n[g]**2)))

def models(panels,settings,pilot,paid):
    result=[Posterior(p,settings[p.name],pilot,paid[p.name]) for p in panels]
    for m in result:m.panel_count=len(panels)
    return result

def gate(settings):
    usable=[s for s in settings.values() if np.isfinite(s['loo_mse_scaled']) and np.isfinite(s['baseline_mse_scaled'])]
    if not usable:return False,np.nan
    numerator=np.mean([s['loo_mse_scaled'] for s in usable]);den=np.mean([s['baseline_mse_scaled'] for s in usable])
    r2=1-numerator/den if den>0 else -np.inf
    return bool(r2>0),float(r2)

def choose(scores,available,priority):
    ii=np.flatnonzero(available);v=scores[ii];best=np.max(v)
    assert np.isfinite(best)
    tied=ii[np.isclose(v,best,rtol=1e-12,atol=1e-14)]
    return int(tied[np.argmin(priority[tied])])

def path(d,panels,reveal,pilot,paid,settings,budgets,design,rng):
    assert not any(c=='y' or c.startswith('y_') for c in d)
    n=len(d);selected=list(map(int,pilot));available=np.ones(n,bool);available[pilot]=False
    priority=np.empty(n,int);priority[rng.permutation(n)]=np.arange(n)
    states=models(panels,settings,pilot,paid)
    snapshots={};on=set(budgets);passed,r2=gate(settings)
    labels=pd.factorize(pd.Series(list(zip(d.year,d.treatment)),dtype=object),sort=False)[0]
    counts=np.bincount(labels[pilot],minlength=int(labels.max())+1)
    X,_=engine.selection_features(d,panels)
    sd=X.std(axis=0);keep=sd>0
    X=(X[:,keep]-X[:,keep].mean(axis=0))/sd[keep] if keep.any() else np.zeros((n,1))
    distances=np.min(np.sum((X[:,None,:]-X[pilot][None,:,:])**2,axis=2),axis=1)
    while True:
        b=len(selected)
        if b in on:
            snapshots[b]=dict(selected=np.array(selected),predictions={m.p.name:m.predictions() for m in states},
                              predicted_risk=sum(m.risk() for m in states),gate_passed=passed,pilot_r2=r2)
        if b==max(budgets):break
        if design=='random':i=int(rng.choice(np.flatnonzero(available)))
        elif design=='cell_balance':i=choose(-counts[labels].astype(float),available,priority)
        elif design=='maximin':i=choose(distances,available,priority)
        elif design=='active_sampling':
            ii=np.flatnonzero(available)
            scores=np.sqrt(np.maximum(sum(m.contribution_second_moment() for m in states),0)) if passed else np.ones(n)
            probs=scores[ii];total=probs.sum()
            if not np.isfinite(total) or total<=0:probs=np.ones(len(ii))/len(ii)
            else:probs=probs/total
            i=int(rng.choice(ii,p=probs))
        else:i=choose(sum(m.scores(design) for m in states),available,priority)
        got=reveal(np.array([i]))
        for m in states:m.update(i,float(got[m.p.name][0]))
        selected.append(i);available[i]=False;counts[labels[i]]+=1
        if design=='maximin':distances=np.minimum(distances,np.sum((X-X[i])**2,axis=1))
    return snapshots

def reconstruct(panels,settings,selected,observed):
    result={};risk=0.
    for p in panels:
        ii=np.asarray(selected)[p.eligible[selected]];z=p.F['year_affine'];zs=z[ii]
        V=np.linalg.inv(settings[p.name]['lam']*np.eye(z.shape[1])+zs.T@zs)
        beta=V@zs.T@(observed[p.name][ii]-p.x[ii])
        f=p.x+z@beta;f[ii]=observed[p.name][ii];result[p.name]=f
    return result

class PaidOracle:
    def __init__(self,responses):
        self.__responses=responses;self.paid=set();self.calls=[]
    def reveal(self,indices):
        ix=np.asarray(indices,int)
        assert len(set(ix))==len(ix) and not self.paid.intersection(map(int,ix))
        self.paid.update(map(int,ix));self.calls.append(ix.tolist())
        return {k:v[ix].copy() for k,v in self.__responses.items()}
