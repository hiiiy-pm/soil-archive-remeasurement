"""Paired equal-budget outcomes for both frozen multiobjective allocation rules."""
import argparse
import json
from pathlib import Path
from statistics import NormalDist

from multiperiod_run import OUT, REPEATS, SEED
from joint_summary import NEW, PRIMARY, decisions, domain_family_weights, mean_and_mcse
from trials import TargetSet, exact_targets, np, pd
from experiment import conditional


Z = NormalDist().inv_cdf(.975)
SIM = NormalDist().inv_cdf(1 - .05 / (2 * 111))
INDEX = ['site', 'model', 'budget', 'group', 'metric']


def read_draws(site, output):
    folder = output / 'benchmarks' / site
    config = json.loads((folder / 'config.json').read_text())
    n = config['records']
    size = len(config['budgets']), len(config['models']), len(config['coordinates']), len(config['metrics'])
    assert config['replicates'] == REPEATS and config['seed'] == SEED and n == config['budgets'][-1]
    assert config['ratio'] == 1 and config['models'] == ['kriging', 'ridge']
    assert len(config['designs']) == 18 and size[0] == 10
    cube = np.empty((REPEATS, 20, *size))
    for rep in range(REPEATS):
        name = f'draw_{rep:03d}.npz'
        with np.load(folder / name) as base, np.load(output / 'fixed' / site / name) as fixed:
            pilot = np.random.default_rng(SEED + rep).choice(n, config['budgets'][0], replace=False)
            np.testing.assert_array_equal(base['pilot'], pilot)
            np.testing.assert_array_equal(fixed['pilot'], pilot)
            for p in fixed['paths']:
                np.testing.assert_array_equal(p[:len(pilot)], pilot)
                np.testing.assert_array_equal(np.sort(p), np.arange(n))
            mask = np.unpackbits(base['selections'], axis=-1, count=n)
            np.testing.assert_array_equal(mask.sum(axis=-1), np.broadcast_to(config['budgets'], mask.shape[:2]))
            assert mask[:, :, pilot].all()
            cube[rep, :18] = base['scores']
            s = fixed['scores']
            cube[rep, 18] = s[0]
            cube[rep, 19] = .5 * (s[1] + s[2])
    assert np.isfinite(cube).all() and not cube[:, :, -1].any()
    np.testing.assert_array_equal(cube[:, :, 0], np.broadcast_to(cube[:, :1, 0], cube[:, :, 0].shape))
    return config, cube


def time_pairs(site, output, cfg, cube, loader):
    """Reconstruct time contrasts directly for all supported campaign pairs.

    Return nine-budget means by named domain and year pair. Recombining the
    contrasts must reproduce the recorded domain-specific error at each
    individual budget for all twenty policies.
    """
    frame, panels, channels, responses, fractions = loader(site)
    specs, detail = [], []
    for channel in channels:
        target = TargetSet(panels, channel)
        truth = np.array([float(v) for v in exact_targets(panels, target, fractions)])
        keep = np.zeros(len(truth), bool)
        for pi, name, family, start, stop, _ in target.groups:
            if family != 'time':
                continue
            keep[start:stop] = True
            for j in range(start, stop):
                _, ids, _ = target.definitions[j]
                yr = sorted(panels[pi].cells[int(i)][0] for i in ids)
                detail.append((name, yr[0], yr[1], truth[j]))
        reference = responses[panels[channel.panel_ids[0]].name][channel.rows]
        cov = channel.kernel + cfg['ratio'] * np.eye(len(channel.x))
        specs.append((channel, cov, reference, target.C[keep], truth[keep]))
    table = pd.DataFrame(detail, columns=['domain','first_year','last_year','truth'])
    assert not table.empty

    def reconstruct(mask):
        scored = []
        for ch, cov, reference, contrasts, truth in specs:
            paid = mask[ch.rows]
            ix = np.flatnonzero(paid)
            pred = ch.x + conditional(cov, ix, reference[ix] - ch.x[ix])
            pred[ix] = reference[ix]
            # Match the scorer's two-column matrix product; only the kriging
            # column enters this diagnostic.
            estimated = (contrasts @ np.column_stack([pred, pred]))[:, 0]
            resolved = ~(contrasts[:, ~paid] != 0).any(axis=1)
            estimated[resolved] = truth[resolved]
            scored.append(np.column_stack((np.sign(estimated) != np.sign(truth), (estimated - truth)**2)))
        return np.concatenate(scored)

    n = len(frame)
    totals = np.zeros((21, len(cfg['budgets']), len(table), 2))
    for i in range(REPEATS):
        filename = f'draw_{i:03d}.npz'
        with np.load(output / 'benchmarks' / site / filename) as base, np.load(output / 'fixed' / site / filename) as fixed:
            masks = np.zeros((21, len(cfg['budgets']), n), bool)
            masks[:18] = np.unpackbits(base['selections'], axis=-1, count=n)
            for d, path in enumerate(fixed['paths'], 18):
                for b, budget in enumerate(cfg['budgets']):
                    masks[d, b, path[:budget]] = True
            cache = {}
            for d, schedule in enumerate(masks):
                for b, mask in enumerate(schedule):
                    key = mask.tobytes()
                    if key not in cache:
                        cache[key] = reconstruct(mask)
                    totals[d, b] += cache[key]
    totals /= REPEATS
    # Each pilot carries two equally probable starting phases of alternation.
    means = np.concatenate([totals[:19], totals[19:21].mean(axis=0, keepdims=True)])
    names = cfg['designs'] + NEW
    for domain in table.domain.unique():
        support = table.domain.eq(domain).to_numpy()
        nonzero = support & table.truth.ne(0).to_numpy()
        coord = cfg['coordinates'].index([domain, 'time'])
        for col, metric, include in [(0, 'direction', nonzero), (1, 'native_mse', support)]:
            recombined = means[:, :, include, col].mean(axis=2)
            original = cube[:, :, :, 0, coord, cfg['metrics'].index(metric)].mean(axis=0)
            np.testing.assert_allclose(recombined, original, rtol=1e-10, atol=1e-11)
    noncensus = means[:, :-1].mean(axis=1)
    rows = []
    for (domain, first, last), grp in table.groupby(['domain','first_year','last_year']):
        ids = grp.index.to_numpy()
        nz = grp.index[grp.truth.ne(0)].to_numpy()
        for d, name in enumerate(names):
            losses = noncensus[d]
            rows.append([site, domain, first, last, len(nz), len(ids)-len(nz), name,
                         losses[nz, 0].mean() if len(nz) else np.nan, losses[ids,1].mean()**.5])
    return pd.DataFrame(rows, columns=['site','domain','first_year','last_year','nonzero','tied',
                                       'design','direction','native_rmse'])


def summarize(site, output, loader=None):
    cfg, cube = read_draws(site, output)
    designs = cfg['designs'] + NEW
    models, metrics, coords = cfg['models'], cfg['metrics'], cfg['coordinates']
    periods = [(str(b), slice(i, i + 1)) for i, b in enumerate(cfg['budgets'])]
    periods.extend([('noncensus', slice(None, -1)), ('after_pilot', slice(1, -1))])
    performance, paired, domains = [], [], []
    for period, periods_slice in periods:
        by_pilot = cube[:, :, periods_slice].mean(axis=2)
        overall = by_pilot.mean(axis=0)
        for d, design in enumerate(designs):
            for m, model in enumerate(models):
                for j, (domain, family) in enumerate(coords):
                    domains.append([site, model, period, domain, family, design, *overall[d, m, j]])
        for group in ('time', 'treatment', 'primary', 'interaction'):
            if group != 'primary' and not any(family == group for _, family in coords):
                continue
            weight = domain_family_weights(coords, group)
            arr = np.einsum('rdmjk,j->rdmk', by_pilot[..., :-1], weight)
            means, errors = mean_and_mcse(arr)
            for m, model in enumerate(models):
                for k, metric in enumerate(metrics[:-1]):
                    key = [site, model, period, group, metric]
                    for d, design in enumerate(designs):
                        mean, se = means[d,m,k], errors[d,m,k]
                        performance.append([*key, design, mean, se, mean-Z*se, mean+Z*se])
                    adjusted = model == 'kriging' and period == 'noncensus' and (group, metric) in PRIMARY
                    for focal in NEW:
                        f = designs.index(focal)
                        diff, sed = mean_and_mcse(arr[:, f:f+1, m, k] - arr[:, :, m, k])
                        for d, other in enumerate(designs):
                            if f == d:
                                continue
                            v, e = diff[d], sed[d]
                            paired.append([*key, focal, other, v, e, v-Z*e, v+Z*e,
                                           v-SIM*e if adjusted else np.nan,
                                           v+SIM*e if adjusted else np.nan])
    perf = pd.DataFrame(performance, columns=INDEX+['design', 'mean', 'mcse', 'lower', 'upper'])
    pairs = pd.DataFrame(paired, columns=INDEX+['focal','comparator','difference','mcse','lower','upper',
                                                'simultaneous_lower','simultaneous_upper'])
    domain = pd.DataFrame(domains, columns=['site','model','budget','domain','family','design',*metrics])
    domain['native_rmse'] = np.sqrt(domain.native_mse)
    adjusted = pairs.dropna(subset=['simultaneous_upper']).copy()
    adjusted['unordered'] = [tuple(sorted(x)) for x in zip(adjusted.focal, adjusted.comparator)]
    assert len(adjusted) == 114 and len(adjusted.drop_duplicates(INDEX+['unordered'])) == 111
    best, replication = decisions(perf, pairs)
    mse = perf[(perf.group == 'primary') & (perf.metric == 'scaled_mse')]
    for i, row in replication.iterrows():
        means = mse[(mse.model == row.model) & (mse.budget == row.budget)].set_index('design')['mean']
        span = means.sign_error - means.contrast_mse
        improvement = means.sign_error - means[row.focal]
        replication.loc[i, 'mse_fraction_lower_than_direct'] = improvement / means.sign_error if means.sign_error else np.nan
        replication.loc[i, 'direct_minus_matched_mse'] = span
        replication.loc[i, 'mse_penalty_fraction_recovered'] = improvement / span if span > 0 else np.nan
        if row.direct_time_gain <= 0:
            replication.loc[i, 'time_gain_fraction_retained'] = np.nan
    folder = output / site
    folder.mkdir(exist_ok=True, parents=True)
    for filename, table in [('performance',perf),('paired',pairs),('domains',domain),
                            ('best_baselines',best),('replication',replication)]:
        table.to_csv(folder / (filename + '.csv'), index=False)
    if loader is not None:
        time_pairs(site, output, cfg, cube, loader).to_csv(folder / 'year_pairs.csv', index=False)
    evaluated = replication[(replication.model == 'kriging') & (replication.budget == 'noncensus')]
    print(evaluated.to_string(index=False))


if __name__ == '__main__':
    from multiperiod_trial import SITE, load_trial

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    args = parser.parse_args()
    summarize(SITE, args.output_dir, load_trial)
