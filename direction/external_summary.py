"""Paired accuracy of pilot-only choice, fixed allocations and a hindsight diagnostic."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
from statistics import NormalDist

from trials import np, pd
from external_run import FIXED, OUT, REPEATS
from joint_summary import domain_family_weights


PRIMARY = [('time', 'direction'), ('treatment', 'direction'), ('primary', 'scaled_mse')]
Z = NormalDist().inv_cdf(0.975)
ZS = NormalDist().inv_cdf(1 - 0.05 / (2 * 60))


def sign_order(fixed, coords, family, nonzero_counts):
    """Exact ordering of nine-budget single-family sign losses for joint vs fair alternation."""
    value = Fraction()
    for i, (domain, target_family) in enumerate(coords):
        if target_family == family:
            n = nonzero_counts[(domain, target_family)]
            # All three individual paths yield an exact integer number wrong.
            raw = fixed[:, :-1, 0, i, 0] * n
            count = np.rint(raw).astype(int)
            np.testing.assert_allclose(raw, count, atol=1e-9, rtol=0)
            totals = count.sum(axis=1)
            value += Fraction(int(2 * totals[0] - totals[1] - totals[2]), int(n))
    return int(value > 0) - int(value < 0)


def load_results(site, output):
    base_folder = output / 'benchmarks' / site
    folder = output / site
    config = json.loads((base_folder / 'config.json').read_text())
    design = config['designs'] + FIXED + ['gate']
    coords = config['coordinates']
    inventory = pd.read_csv(base_folder / 'contrast_inventory.csv')
    counts = inventory[inventory.true_sign.ne(0)].groupby(['domain', 'family']).size()
    assert len(design) == 21 and config['replicates'] == REPEATS and len(config['budgets']) == 10
    assert config['models'] == ['kriging', 'ridge']
    cube = np.empty((REPEATS, len(design), 10, 2, len(coords), len(config['metrics'])))
    decisions, orders = [], []
    for rep in range(REPEATS):
        name = f'draw_{rep:03d}.npz'
        with (np.load(base_folder / name) as base,
              np.load(output / 'fixed' / site / name) as fixed,
              np.load(folder / 'choices' / name) as gate,
              np.load(folder / 'scores' / name) as observed):
            n = config['records']
            for array in (fixed, gate, observed):
                np.testing.assert_array_equal(base['pilot'], array['pilot'])
            mask = np.unpackbits(base['selections'], axis=-1, count=n)
            np.testing.assert_array_equal(mask.sum(axis=-1),
                                          np.broadcast_to(config['budgets'], mask.shape[:2]))
            assert mask[:, :, base['pilot']].all()
            for path in fixed['paths']:
                np.testing.assert_array_equal(path[:len(base['pilot'])], base['pilot'])
                np.testing.assert_array_equal(np.sort(path), np.arange(n))
            choice = str(gate['choice'])
            assert choice in FIXED and str(observed['choice']) == choice
            arms = fixed['scores']
            cube[rep, :18] = base['scores']
            cube[rep, 18] = arms[0]
            cube[rep, 19] = .5 * (arms[1] + arms[2])
            cube[rep, 20] = observed['scores']
            np.testing.assert_array_equal(cube[rep, 20], cube[rep, design.index(choice)])
            gain = gate['gains']
            frac = np.divide(gain, gain.max(axis=0), out=np.ones_like(gain), where=gain.max(axis=0)>0)
            quality = frac.min(axis=1)
            best = FIXED[0] if quality[0] > quality[1] and not np.isclose(
                quality[0], quality[1], rtol=1e-12, atol=1e-15) else FIXED[1]
            assert choice == best
            decisions.append(dict(site=site, repeat=rep, choice=choice,
                                  **{f'{arm}_{k}_gain':gain[i,j] for i,arm in enumerate(FIXED)
                                     for j,k in enumerate(['time_sign','treatment_sign','combined_mse'])}))
            orders.append([sign_order(arms, coords, family, counts) for family in ('time', 'treatment')])
    assert np.isfinite(cube).all() and not cube[:, :, -1].any()
    np.testing.assert_allclose(cube[:, :, 0], np.broadcast_to(cube[:, :1, 0], cube[:, :, 0].shape))
    return config, design, cube, pd.DataFrame(decisions), np.array(orders)


def summarize(site, output):
    cfg, design, cube, decisions, orders = load_results(site, output)
    coords, models, metrics = cfg['coordinates'], cfg['models'], cfg['metrics']
    periods = [(str(b), [i]) for i, b in enumerate(cfg['budgets'])]
    periods += [('noncensus', slice(None,-1)), ('after_pilot', slice(1,-1))]
    performance, paired, domains, best, conditional, decision_quality, observed_pilots = [], [], [], [], [], [], []
    chose_joint = decisions.choice.eq(FIXED[0]).to_numpy()
    for period, ix in periods:
        # Collapse each replicate's budgets before estimating variances over pilots.
        draws = cube[:, :, ix].mean(axis=2)
        coord_means = draws.mean(axis=0)
        for di, rule in enumerate(design):
            for mi, model in enumerate(models):
                for ci, (domain, family) in enumerate(coords):
                    metrics_row = dict(zip(metrics,coord_means[di,mi,ci]))
                    domains.append(dict(site=site,budget=period,model=model,domain=domain,family=family,
                                        design=rule,**metrics_row,native_rmse=metrics_row['native_mse']**.5))
        for group in ('time', 'treatment', 'primary', 'interaction'):
            if group != 'primary' and not any(f == group for _,f in coords):
                continue
            w = domain_family_weights(coords,group)
            arr = np.einsum('rdmjk,j->rdmk',draws[..., :-1],w)
            means = arr.mean(axis=0)
            sem = arr.std(axis=0,ddof=1) / REPEATS**.5
            diff = arr[:,-1:] - arr[:,:-1]
            delta = diff.mean(axis=0)
            delta_se = diff.std(axis=0,ddof=1) / REPEATS**.5
            for mi, model in enumerate(models):
                for k, metric in enumerate(metrics[:-1]):
                    ident = dict(site=site, model=model, budget=period, group=group, metric=metric)
                    main = model=='kriging' and period=='noncensus' and (group,metric) in PRIMARY
                    for di, rule in enumerate(design):
                        avg, err = means[di,mi,k], sem[di,mi,k]
                        performance.append(dict(**ident,design=rule,mean=avg,mcse=err,
                                                lower=avg-Z*err,upper=avg+Z*err))
                        if rule != 'gate':
                            d, e = delta[di,mi,k], delta_se[di,mi,k]
                            paired.append(dict(**ident,focal='gate',comparator=rule,difference=d,mcse=e,
                                               lower=d-Z*e,upper=d+Z*e,
                                               simultaneous_lower=d-ZS*e if main else np.nan,
                                               simultaneous_upper=d+ZS*e if main else np.nan))
                    if not main:
                        continue
                    scores = arr[:, -3:, mi, k]
                    gaps = scores[:,0] - scores[:,1]
                    idx = PRIMARY.index((group,metric))
                    order = orders[:,idx] if idx < 2 else np.sign(gaps)
                    if idx < 2:
                        assert (np.abs(gaps[order==0]) < 1e-12).all()
                        assert (np.sign(gaps[order!=0])==order[order!=0]).all()
                    direction = np.where(chose_joint, order, -order)
                    oracle = np.min(scores[:,:2],axis=1)
                    index = int(np.argmin(means[:18,mi,k]))
                    best.append(dict(**ident,best_original=design[index],original_mean=means[index,mi,k],
                                     gate_mean=means[-1,mi,k],difference=delta[index,mi,k],
                                     simultaneous_lower=delta[index,mi,k]-ZS*delta_se[index,mi,k],
                                     simultaneous_upper=delta[index,mi,k]+ZS*delta_se[index,mi,k]))
                    decision_quality.append(dict(**ident,total=REPEATS,
                        chosen_strictly_better=int(np.sum(direction<0)), chosen_tied=int(np.sum(direction==0)),
                        chosen_worse=int(np.sum(direction>0)),
                        always_joint_strictly_better=int(np.sum(order<0)),
                        always_alternate_strictly_better=int(np.sum(order>0)),
                        oracle_mean=oracle.mean(),
                        gate_mean=scores[:,2].mean(),oracle_gap=(scores[:,2]-oracle).mean()))
                    for rule, take in zip(FIXED,(chose_joint,~chose_joint)):
                        if take.any():
                            conditional.append(dict(**ident,choice=rule,pilots=int(take.sum()),
                                joint_mean=scores[take,0].mean(),alternate_mean=scores[take,1].mean(),
                                gate_mean=scores[take,2].mean(),chosen_better=int((direction[take]<0).sum()),
                                chosen_tied=int((direction[take]==0).sum()),chosen_worse=int((direction[take]>0).sum())))
                    for rep in range(REPEATS):
                        observed_pilots.append(dict(**ident,repeat=rep,choice=decisions.choice.iloc[rep],
                            joint=scores[rep,0],alternate=scores[rep,1],gate=scores[rep,2],oracle=oracle[rep],
                            gate_vs_other_order=int(direction[rep])))
    folder = output / site
    decisions.to_csv(folder / 'pilot_choices.csv',index=False)
    assert sum(pd.notna(r['simultaneous_upper']) for r in paired) == 60
    for name, rows in [('performance',performance),('paired',paired),('domains',domains),
                       ('best_baselines',best),('decision_quality',decision_quality),
                       ('conditional',conditional),('pilot_outcomes',observed_pilots)]:
        pd.DataFrame(rows).to_csv(folder / (name+'.csv'),index=False)
    print(decisions.choice.value_counts().to_string(),flush=True)
    print(pd.DataFrame(decision_quality).to_string(index=False),flush=True)


if __name__=='__main__':
    from independent_trial import SITE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    args=parser.parse_args()
    summarize(SITE,args.output_dir)
