"""Replay both fixed joint strategies and eighteen baselines on the independent archive."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joint_run
import study
from trials import ROOT


OUT = ROOT / 'direction' / 'multiperiod_results'
SEED = 1062719
REPEATS = 200


def run_chunk(site, start, stop, output):
    from multiperiod_trial import load_trial

    study.run(site, start, stop, output_root=output / 'benchmarks', loader=load_trial,
              trial_seed=SEED)
    joint_run.run(site, start, stop, output_root=output / 'fixed',
                  baseline_root=output / 'benchmarks', loader=load_trial)
    return start, stop


def main():
    from multiperiod_trial import SITE, load_trial

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int, default=REPEATS)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    args = parser.parse_args()
    if not (0 <= args.start < args.stop <= REPEATS and args.workers > 0):
        parser.error('Specify a nonempty subset of the 200 pilots and a positive worker count.')
    study.run(SITE, 0, 0, output_root=args.output_dir / 'benchmarks', loader=load_trial,
              trial_seed=SEED)
    if args.workers == 1:
        run_chunk(SITE, args.start, args.stop, args.output_dir)
        return
    # Each worker uses single-threaded BLAS; batches publish checkpoints for
    # both sets of strategies and independently resume from existing draw files.
    step = max(1, min(5, (args.stop - args.start + args.workers - 1) // args.workers))
    with ProcessPoolExecutor(max_workers=min(args.workers, args.stop - args.start)) as pool:
        jobs = [pool.submit(run_chunk, SITE, lo, min(lo + step, args.stop), args.output_dir)
                for lo in range(args.start, args.stop, step)]
        for job in as_completed(jobs):
            lo, hi = job.result()
            print('Verified equal-budget draws', lo, 'through', hi - 1, flush=True)


if __name__ == '__main__':
    main()
