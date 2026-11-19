"""Reassess two released external soil trials with a common, plot-aware calibration.

Run with the original reproduction package on disk. Set SOIL_REPRO to its
extracted directory when using a separate data-package root.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PACKAGE = Path(os.environ.get("SOIL_REPRO", HERE.parent))
sys.path.insert(0, str(PACKAGE / "code"))
import pilot_risk as original

OBJECTIVES = ("retained_contrast", "concentration", "conversion")
NUGGETS = 10.0 ** np.arange(-4, 5)

def discrepancy_kernel(frame, x):
    scaled = (x[:, None] - x) / x.std()
    components = [np.exp(-0.5 * scaled**2)]
    for key in ["plot", "treatment", "year", "depth"]:
        labels = frame[key].to_numpy()
        assert not pd.isna(labels).any()
        if len(np.unique(labels)) > 1:
            components.append(labels[:, None] == labels)
    return np.mean(components, axis=0)

def pilot_noise(kernel, pilot, delta):
    block = kernel[np.ix_(pilot, pilot)]
    losses = []
    for nugget in NUGGETS:
        inverse = np.linalg.inv(block + nugget * np.eye(len(pilot)))
        sums = inverse.sum(axis=1)
        precision = inverse - np.outer(sums, sums) / sums.sum()
        errors = (precision @ delta) / np.diag(precision)
        losses.append(np.mean(errors**2))
    return NUGGETS[np.argmin(losses)]

def conditional(cov, selected, delta, need_cov=False):
    cross = cov[:, selected]
    block = cross[selected]
    solutions = np.linalg.solve(block, np.column_stack([delta, np.ones(len(delta))]))
    intercept = solutions[:, 0].sum() / solutions[:, 1].sum()
    estimate = cross @ (solutions[:, 0] - intercept * solutions[:, 1]) + intercept
    if not need_cov:
        return estimate
    remaining_mean = 1 - cross @ solutions[:, 1]
    variance = cov - cross @ np.linalg.solve(block, cross.T)
    variance += np.outer(remaining_mean, remaining_mean) / solutions[:, 1].sum()
    return estimate, variance

class Targets:
    def __init__(self, panels, size):
        rows, weights, groups, coordinates = [], [], [], []
        self.concentration_weight = np.zeros(size)
        for p in panels:
            aggregation = np.zeros((len(p.n), size))
            aggregation[p.g[p.rows], p.rows] = 1 / p.n[p.g[p.rows]]
            self.concentration_weight[p.rows] = 1 / (len(panels)*len(p.rows)*p.scale)
            for family, (indices, coefficients) in p.K.items():
                contrast = np.einsum('ijk,ij->ik', aggregation[indices], coefficients)
                groups.append((len(rows), len(rows) + len(contrast), p.scale))
                coordinates.append((p.name, family))
                rows.extend(contrast)
                weights.extend([1 / (len(panels)*len(p.K)*len(contrast)*p.scale)] * len(contrast))
        self.matrix = np.asarray(rows)
        self.groups = groups
        self.coordinates = coordinates
        self.joint = np.array([1/(len(panels)*len(p.K)) for p in panels for _ in p.K])
        weighted = self.matrix * np.sqrt(weights)[:, None]
        # Compact factor with the same quadratic loss.
        _, singular, vt = np.linalg.svd(weighted, full_matrices=False)
        keep = singular > singular[0]*1e-12
        self.factor = singular[keep,None]*vt[keep]

    def evaluate(self, prediction, target):
        predicted = self.matrix @ prediction
        result = []
        for start, end, scale in self.groups:
            truth = target[start:end]
            estimates = predicted[start:end]
            nonzero = np.abs(truth)>1e-12
            wrong = (estimates*truth < 0) | (np.abs(estimates)<=1e-12)
            result.append([np.mean((estimates-truth)**2)/scale, wrong[nonzero].mean()])
        return np.asarray(result)

def acquire(kernel, targets, nugget, pilot, priority_seed, objective):
    # This function receives no reference values. The paid pilot supplies nugget.
    n = len(kernel)
    cov = kernel + nugget*np.eye(n)
    _, remaining = conditional(cov, pilot, np.zeros(len(pilot)), True)
    if objective == "conversion":
        # Covariance of latent functionals and observable records, given pilot.
        cross = targets.factor @ kernel
        ksp = cov[np.ix_(pilot,pilot)]
        solved = np.linalg.solve(ksp,cov[pilot,:])
        q = np.linalg.solve(ksp,np.ones(len(pilot)))
        uncertainty = cross - cross[:,pilot] @ solved
        uncertainty += np.outer(targets.factor.sum(axis=1)-cross[:,pilot]@q,1-solved.sum(axis=0))/q.sum()
    else:
        uncertainty = targets.factor @ remaining
    rng = np.random.default_rng(priority_seed)
    priority = np.empty(n, int)
    priority[rng.permutation(n)] = np.arange(n)
    available = np.ones(n,dtype=bool)
    available[pilot] = False
    path = list(pilot)
    while np.any(available):
        ids = np.flatnonzero(available)
        den = np.diag(remaining)[ids]
        assert den.min()>0
        if objective=="concentration":
            numerator = targets.concentration_weight @ (remaining[:,ids]**2)
        else:
            numerator = np.sum(uncertainty[:,ids]**2,axis=0)
        scores = numerator/den
        tied = ids[np.isclose(scores,scores.max(),rtol=1e-12,atol=1e-14)]
        i = tied[np.argmin(priority[tied])]
        v = remaining[:,i].copy()
        row = v / v[i]
        remaining -= np.outer(v,row)
        remaining[i,:]=0
        remaining[:,i]=0
        uncertainty -= np.outer(uncertainty[:,i],row)
        path.append(i)
        available[i]=False
    assert len(set(path))==n
    return np.asarray(path)

def old_ridge(panels, settings, selected, paid_y):
    answer = np.zeros(len(panels[0].x))
    for p in panels:
        use = p.eligible[selected]
        ix = selected[use]
        z = p.F['year_affine']
        data = z[ix]
        coeff = np.linalg.solve(settings[p.name]['lam']*np.eye(z.shape[1])+data.T@data,
                                data.T@(paid_y[use]-p.x[ix]))
        answer[p.rows] = p.x[p.rows] + z[p.rows]@coeff
    answer[selected]=paid_y
    return answer

def run(site):
    folder = PACKAGE/'results'/site
    config = json.loads((folder/'config.json').read_text())
    df, panels, responses = original.load(site)
    assert np.all(np.sum([p.eligible for p in panels], axis=0)==1)
    x = sum(p.x for p in panels)
    y = next(iter(responses.values()))
    assert all(np.array_equal(y, a) for a in responses.values())
    kernel = discrepancy_kernel(df, x)
    targets = Targets(panels, len(df))
    assert [list(q) for q in targets.coordinates]==config['coordinates']
    truth = targets.matrix @ y
    budgets, existing = config['budgets'], config['designs']
    designs = existing + ['pooled_'+d for d in OBJECTIVES]
    scores = np.empty((200,len(designs),len(budgets),2,len(targets.groups),2))
    selections = np.empty((200,len(OBJECTIVES),len(df)),int)
    nuggets=[]
    for rep in range(200):
        with np.load(folder/f'draw_{rep:04d}.npz') as draw:
            pilot=draw['pilot']
            nugget=pilot_noise(kernel,pilot,y[pilot]-x[pilot])
            nuggets.append(nugget)
            cov=kernel+nugget*np.eye(len(df))
            settings={p.name:original.pilot_settings(p,pilot,y[pilot]) for p in panels}
            for di,design in enumerate(designs):
                if di>=len(existing):
                    kind=design.removeprefix('pooled_')
                    seed=config['seed_base']+config['universe_seed_offset']+100000000+existing.index(kind)*10000+rep
                    path=acquire(kernel,targets,nugget,pilot,seed,kind)
                    selections[rep,di-len(existing)]=path
                else:
                    path=draw['path__'+design] if 'path__'+design in draw else None
                for bi,budget in enumerate(budgets):
                    ix=path[:budget] if path is not None else draw[f'indices__{design}__{budget}']
                    assert len(set(ix))==budget and set(pilot)<=set(ix)
                    paid=y[ix]
                    if di<len(existing):
                        saved=draw['scores'][di,bi,0]
                        scores[rep,di,bi,0]=saved[:,[4,1]]
                    if di>=len(existing) or (rep==0 and design=='random'):
                        pred=old_ridge(panels,settings,ix,paid)
                        raw=targets.evaluate(pred,truth)
                        if di<len(existing):
                            np.testing.assert_allclose(raw,scores[rep,di,bi,0],atol=1e-10)
                        else:
                            scores[rep,di,bi,0]=raw
                    pred=x+conditional(cov,ix,paid-x[ix])
                    pred[ix]=paid
                    scores[rep,di,bi,1]=targets.evaluate(pred,truth)
        if rep%50==0: print(site,rep,flush=True)
    assert np.max(np.abs(scores[:,:,-1]))<1e-10
    np.testing.assert_allclose(scores[:,:,0],np.broadcast_to(scores[:,0:1,0],scores[:,:,0].shape),atol=1e-8)
    np.savez_compressed(HERE/(site+'.npz'),scores=scores,paths=selections,nuggets=nuggets,
                        budgets=budgets,designs=designs,coordinates=targets.coordinates,
                        reconstructions=['original_ridge','pooled_kriging'],metrics=['scaled_mse','direction'],
                        joint_weights=targets.joint)
    print(site,'complete',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('site',choices=['beltsville','lincoln'])
    run(parser.parse_args().site)
