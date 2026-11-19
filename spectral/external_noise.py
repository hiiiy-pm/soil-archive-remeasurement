"""Learn a single regularization strength away from the three evaluation sites."""
from experiment import (HERE, PACKAGE, NUGGETS, OBJECTIVES, Targets, acquire, conditional,
                        discrepancy_kernel, old_ridge, original)
import argparse
import json
import numpy as np
import pandas as pd

TRAIN = ('KBS', 'Mandan')
DATA = PACKAGE / 'data/source/Spectral_SOC/measurements_and_spectra.csv'
TEST_SEED = 583771
PENDLETON_RULES = ['random', 'cell_balance', 'maximin', 'concentration', 'conversion', 'retained_contrast']

def public_pairs(site):
    columns = ['Woodwell.ID', 'Trial', 'Year', 'U_depth (cm)', 'L_depth', 'Block.treatment',
               'soc.observed', 'mean.soc.eml.woodwell']
    data = pd.read_csv(DATA, usecols=columns, dtype=str, keep_default_na=False)
    subset = data[data.Trial.eq(site)].copy()
    for field in ['soc.observed', 'mean.soc.eml.woodwell']:
        subset[field] = pd.to_numeric(subset[field], errors='coerce')
    subset = subset.dropna(subset=['soc.observed', 'mean.soc.eml.woodwell']).reset_index(drop=True)
    assert subset['Woodwell.ID'].is_unique
    frame = pd.DataFrame(dict(year=subset.Year.astype(int), treatment=subset['Block.treatment'],
                              depth=subset['U_depth (cm)']+'_'+subset['L_depth'],
                              plot='unavailable', record_id=subset['Woodwell.ID']))
    return frame, subset['mean.soc.eml.woodwell'].to_numpy()*10, subset['soc.observed'].to_numpy()*10

def train():
    table = []
    for site in TRAIN:
        frame, x, y = public_pairs(site)
        delta = y-x
        kernel = discrepancy_kernel(frame, x)
        for ratio in NUGGETS:
            cov = kernel + np.eye(len(x))*ratio
            solution = np.linalg.solve(cov, np.column_stack([delta, np.ones(len(delta))]))
            information = solution[:, 1].sum()
            intercept = solution[:, 0].sum()/information
            rss = (delta-intercept) @ (solution[:, 0]-intercept*solution[:, 1])
            objective = np.linalg.slogdet(cov)[1] + np.log(information) + (len(x)-1)*np.log(rss/(len(x)-1))
            table.append(dict(site=site, n=len(x), ratio=ratio, restricted_deviance=objective))
    result = pd.DataFrame(table)
    result.to_csv(HERE/'external_training.csv', index=False)
    ratio = result.groupby('ratio').restricted_deviance.sum().idxmin()
    (HERE/'external_parameter.json').write_text(json.dumps(dict(training=list(TRAIN), ratio=float(ratio))))
    print('Training counts', result.groupby('site').n.first().to_dict(), 'ratio',ratio)

def run(site):
    external = json.loads((HERE/'external_parameter.json').read_text())
    assert external['training'] == list(TRAIN)
    ratio = external['ratio']
    local = site == 'pendleton'
    if local:
        frame, x, y = public_pairs('Pendleton')
        depths = sorted(frame.depth.unique())
        panels = [original.engine.make_panel('pendleton_depth_'+depth,frame,x,frame.depth.eq(depth),
                                             'g/kg',len(depths)) for depth in depths]
        assert sum(len(p.rows) for p in panels) == len(x) == 264
        assert all(len(p.K) > 0 for p in panels)
        budgets = np.rint(np.arange(1,11)*len(x)/10).astype(int).tolist()
        old_rules = PENDLETON_RULES
    else:
        folder = PACKAGE/'results'/site
        config = json.loads((folder/'config.json').read_text())
        frame, panels, responses = original.load(site)
        x = sum(p.x for p in panels)
        y = next(iter(responses.values()))
        budgets, old_rules = config['budgets'], config['designs']
    kernel = discrepancy_kernel(frame, x)
    cov = kernel + ratio*np.eye(len(x))
    targets = Targets(panels,len(x))
    truth = targets.matrix @ y
    names = list(old_rules) + ['external_'+d for d in OBJECTIVES]
    scores = np.empty((200,len(names),len(budgets),2,len(targets.groups),2))
    selected_paths = np.empty((200,len(OBJECTIVES),len(x)), int)
    for rep in range(200):
        if local:
            pilot = np.random.default_rng(TEST_SEED+rep).choice(len(x),budgets[0],replace=False)
        else:
            draw = np.load(folder/f'draw_{rep:04d}.npz')
            pilot = draw['pilot']
        settings = {p.name:original.pilot_settings(p,pilot,y[pilot]) for p in panels}
        def reveal(ids):
            return {p.name:y[ids] for p in panels}
        for di, design in enumerate(names):
            kind = design.removeprefix('external_')
            if local:
                seed = TEST_SEED + 100000000 + old_rules.index(kind)*10000 + rep
            else:
                seed = config['seed_base'] + config['universe_seed_offset'] + 100000000 + old_rules.index(kind)*10000 + rep
            if design.startswith('external_'):
                path = acquire(kernel,targets,ratio,pilot,seed,kind)
                selected_paths[rep,di-len(old_rules)] = path
            elif local:
                snapshots = original.path(frame,panels,reveal,pilot,reveal(pilot),settings,budgets,design,
                                          np.random.default_rng(seed))
                path = snapshots[budgets[-1]]['selected']
                assert all(np.array_equal(path[:b],snapshots[b]['selected']) for b in budgets)
            else:
                path = draw['path__'+design] if 'path__'+design in draw else None
            for bi,b in enumerate(budgets):
                ix = path[:b] if path is not None else draw[f'indices__{design}__{b}']
                assert len(set(ix)) == b and set(pilot) <= set(ix)
                if not local and di < len(old_rules):
                    scores[rep,di,bi,0] = draw['scores'][di,bi,0][:,[4,1]]
                else:
                    pred = old_ridge(panels,settings,ix,y[ix])
                    scores[rep,di,bi,0] = targets.evaluate(pred,truth)
                pred = x + conditional(cov,ix,y[ix]-x[ix])
                pred[ix] = y[ix]
                scores[rep,di,bi,1] = targets.evaluate(pred,truth)
        if not local:
            draw.close()
        if rep%50==0:
            print(site,rep,flush=True)
    np.testing.assert_allclose(scores[:,:,0],np.broadcast_to(scores[:,0:1,0],scores[:,:,0].shape),atol=1e-9)
    assert np.max(np.abs(scores[:,:,-1])) < 1e-9
    assert np.isfinite(scores).all()
    np.savez_compressed(HERE/f'external_{site}.npz',scores=scores,paths=selected_paths,
                        ratio=ratio,designs=names,coordinates=targets.coordinates,joint_weights=targets.joint,
                        budgets=budgets,metrics=['scaled_mse','direction'],reconstructions=['original_ridge','external_kriging'])
    print(site,'complete',flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=['train','beltsville','lincoln','pendleton'])
    task = parser.parse_args().task
    if task == 'train':
        train()
    else:
        run(task)
