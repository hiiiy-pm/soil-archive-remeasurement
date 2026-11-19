"""Replay the locked pilot-choice experiment. Run before external_summary.py."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from trials import ROOT, TargetSet, np
import joint_run
import study


OUT = ROOT / 'direction' / 'external_results'
SEED = 962173
REPEATS = 200
FIXED = ['joint_maximin', 'alternate']


def store_draw(path, **arrays):
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(path)


def compute_choices(site, start, stop, output):
    from independent_trial import load_trial
    from pilot_choice import pilot_choice

    frame, panels, channels, references, _ = load_trial(site)
    targets = [TargetSet(panels, ch) for ch in channels]
    config = json.loads((output / 'benchmarks' / site / 'config.json').read_text())
    folder = output / site / 'choices'
    folder.mkdir(parents=True, exist_ok=True)
    for rep in range(start, stop):
        dest = folder / f'draw_{rep:03d}.npz'
        if dest.exists():
            continue
        pilot = np.random.default_rng(SEED + rep).choice(len(frame), config['budgets'][0], replace=False)
        paid = {name: values[pilot].copy() for name, values in references.items()}
        choice, gains = pilot_choice(frame, panels, channels, targets, config['ratio'], pilot,
                                     paid, SEED + 100000 + rep)
        gains = np.asarray(gains)
        assert choice in FIXED and gains.shape == (2, 3) and np.isfinite(gains).all()
        store_draw(dest, pilot=pilot, choice=choice, gains=gains)


def evaluate_choices(site, start, stop, output):
    folder = output / site
    dest = folder / 'scores'
    dest.mkdir(exist_ok=True, parents=True)
    for rep in range(start, stop):
        name = f'draw_{rep:03d}.npz'
        with np.load(folder / 'choices' / name) as decision, np.load(output / 'fixed' / site / name) as fixed:
            np.testing.assert_array_equal(decision['pilot'], fixed['pilot'])
            choice = str(decision['choice'])
            raw = fixed['scores']
            scores = raw[0] if choice == FIXED[0] else (raw[1] + raw[2]) / 2
            store_draw(dest / name, scores=scores, pilot=decision['pilot'], choice=choice)


def work(args):
    from independent_trial import load_trial

    site, start, stop, output, stage = args
    if stage in ['all', 'gate']:
        compute_choices(site, start, stop, output)
    if stage in ['all', 'benchmarks']:
        study.run(site, start, stop, output_root=output / 'benchmarks', loader=load_trial, trial_seed=SEED)
    if stage in ['all', 'fixed']:
        joint_run.run(site, start, stop, output / 'fixed', baseline_root=output / 'benchmarks',
                      loader=load_trial)
    if stage in ['all', 'score']:
        evaluate_choices(site, start, stop, output)
    return start, stop


def main():
    from independent_trial import SITE, load_trial

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all', 'gate', 'benchmarks', 'fixed', 'score'], default='all')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int, default=REPEATS)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    args = parser.parse_args()
    if not (0 <= args.start < args.stop <= REPEATS and args.workers > 0):
        parser.error('Expected a nonempty repeat range from 0 to 200 and positive workers.')
    config = args.output_dir / 'benchmarks' / SITE / 'config.json'
    if not config.exists():
        study.run(SITE, 0, 0, output_root=args.output_dir / 'benchmarks', loader=load_trial, trial_seed=SEED)
    step = max(1, (args.stop - args.start + args.workers - 1) // args.workers)
    jobs = [(SITE, pos, min(pos + step, args.stop), args.output_dir, args.stage)
            for pos in range(args.start, args.stop, step)]
    if args.workers == 1:
        work(jobs[0])
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
            for start, stop in pool.map(work, jobs):
                print('Completed repeats', start, 'through', stop - 1, flush=True)


if __name__ == '__main__':
    main()
