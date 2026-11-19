"""Fixed compromise allocations for paid soil archives."""
import argparse
import json
from pathlib import Path

from acquisition import PosteriorState
from study import Evaluator, METRICS, MODELS, OUT, SEEDS
from trials import ROOT, TargetSet, load_trial, np, original
from experiment import conditional


POLICIES = ('maximin', 'alternate_sign_first', 'alternate_mse_first')
COMPONENTS = ('sign_variance', 'contrast_mse')
JOINT_OUT = ROOT / 'direction' / 'joint_results'


def tied_best(ids, values):
    return ids[np.isclose(values, values.max(), rtol=1e-12, atol=1e-15)]


def choose(ids, gains, priority, component=None):
    """Use either the designated component or the normalized maximin criterion."""
    if component is not None:
        tied = tied_best(ids, gains[component, ids])
    else:
        eligible = gains[:, ids]
        maxima = eligible.max(axis=1, keepdims=True)
        fractions = np.divide(eligible, maxima, out=np.ones_like(eligible), where=maxima > 0)
        positions = tied_best(np.arange(len(ids)), fractions.min(axis=0))
        tied = tied_best(ids[positions], fractions[:, positions].sum(axis=0))
    return tied[np.argmin(priority[tied])]


def joint_path(size, panels, channels, targets, ratio, pilot, paid, reveal, policy, seed):
    states = []
    for channel, target in zip(channels, targets):
        observed = paid[panels[channel.panel_ids[0]].name]
        states.append([PosteriorState(channel, target, ratio, pilot, observed, rule)
                       for rule in COMPONENTS])
    local_ids = [{v:j for j, v in enumerate(c.rows)} for c in channels]
    priority = np.empty(size, int)
    priority[np.random.default_rng(seed).permutation(size)] = np.arange(size)
    available = np.ones(size, bool)
    available[pilot] = False
    path = list(pilot)
    while available.any():
        component = None
        if policy != 'maximin':
            first = int(policy == 'alternate_mse_first')
            component = (first + len(path) - len(pilot)) % 2
        gains = np.zeros((2, size))
        for states_for_channel in states:
            for k in ((0, 1) if component is None else (component,)):
                rows, gain = states_for_channel[k].gains()
                gains[k, rows] += gain
        choice = choose(np.flatnonzero(available), gains, priority, component)
        measurement = reveal(np.array([choice]))
        for channel, local, pair in zip(channels, local_ids, states):
            if choice in local:
                value = measurement[panels[channel.panel_ids[0]].name][0]
                for state in pair:
                    state.update(local[choice], value)
        available[choice] = False
        path.append(choice)
    return np.asarray(path, dtype=np.int16)


def score_set(ix, panels, channels, targets, covs, settings, responses, evaluator, size):
    # Only sorted, measured responses enter either reconstruction; the evaluator
    # compares archived truth after the sample set and predictions are fixed.
    mask = np.zeros(size, bool)
    mask[ix] = True
    indices = np.flatnonzero(mask)
    observed = {key:np.where(mask, values, np.nan) for key, values in responses.items()}
    ridge = original.reconstruct(panels, settings, indices, observed)
    gp = []
    for channel, cov in zip(channels, covs):
        selected = np.flatnonzero(mask[channel.rows])
        values = observed[panels[channel.panel_ids[0]].name][channel.rows[selected]]
        pred = channel.x + conditional(cov, selected, values - channel.x[selected])
        pred[selected] = values
        gp.append(pred)
    return evaluator.score(channels, panels, targets, indices, gp, ridge)


def run(site, start, stop, output_root=JOINT_OUT, *, baseline_root=OUT, loader=load_trial):
    folder = output_root / site
    folder.mkdir(exist_ok=True, parents=True)
    original_folder = baseline_root / site
    config = json.loads((original_folder / 'config.json').read_text())
    frame, panels, channels, responses, fractions = loader(site)
    size = len(frame)
    targets = [TargetSet(panels, channel) for channel in channels]
    evaluator = Evaluator(panels, channels, targets, fractions)
    coordinates = [[d, f] for target in targets for _, d, f, *_ in target.groups]
    assert coordinates == config['coordinates']
    assert METRICS == config['metrics'] and MODELS == config['models']
    budgets = config['budgets']
    ratio = config['ratio']
    covs = [c.kernel + ratio*np.eye(len(c.x)) for c in channels]
    assert size == config['records'] and budgets[-1] == size
    for rep in range(start, stop):
        dest = folder / f'draw_{rep:03d}.npz'
        if dest.exists():
            continue
        pilot = np.random.default_rng(config['seed'] + rep).choice(size, budgets[0], replace=False)
        paid = {key:v[pilot] for key, v in responses.items()}
        settings = {p.name:original.pilot_settings(p, pilot, paid[p.name]) for p in panels}
        scores = np.empty((3, len(budgets), len(MODELS), len(coordinates), len(METRICS)))
        paths = []
        for k, policy in enumerate(POLICIES):
            oracle = original.PaidOracle(responses)
            path = joint_path(size, panels, channels, targets, ratio, pilot, oracle.reveal(pilot),
                              oracle.reveal, policy, config['seed'] + 100000 + rep)
            np.testing.assert_array_equal(path[:len(pilot)], pilot)
            np.testing.assert_array_equal(np.sort(path), np.arange(size))
            assert len(oracle.paid) == size
            paths.append(path)
            for b, budget in enumerate(budgets):
                scores[k,b] = score_set(path[:budget], panels, channels, targets, covs, settings,
                                        responses, evaluator, size)
        assert np.isfinite(scores).all() and np.all(scores[:, -1] == 0)
        with np.load(original_folder / dest.name) as old:
            np.testing.assert_array_equal(old['pilot'], pilot)
            baseline = old['scores']
            for k in range(3):
                np.testing.assert_array_equal(scores[k, 0], baseline[0, 0])
            for k, rule in enumerate(COMPONENTS):
                assert paths[k+1][len(pilot)] == old['path_' + rule][len(pilot)]
                if rep == 0:
                    # Verify scoring on a published nontrivial physical sample set.
                    b = len(budgets)//2
                    ix = old['path_' + rule][:budgets[b]]
                    ref = score_set(ix, panels, channels, targets, covs, settings,
                                    responses, evaluator, size)
                    np.testing.assert_array_equal(ref, baseline[config['designs'].index(rule), b])
        tmp = dest.with_suffix('.tmp')
        with tmp.open('wb') as f:
            np.savez_compressed(f, scores=scores, paths=np.stack(paths), pilot=pilot)
        tmp.replace(dest)
        if rep % 10 == 0:
            print(site, rep, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('site', choices=SEEDS)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int, default=200)
    parser.add_argument('--output-dir', type=Path, default=JOINT_OUT)
    args = parser.parse_args()
    if not 0 <= args.start < args.stop <= 200:
        parser.error('Expected 0 <= start < stop <= 200')
    run(args.site, args.start, args.stop, args.output_dir)
