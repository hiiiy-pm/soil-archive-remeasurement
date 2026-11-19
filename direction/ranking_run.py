"""Replay the fixed direction-family allocations with the original common pilots."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

from acquisition import PosteriorState, sign_gain
from joint_run import choose, score_set
from study import Evaluator, MODELS, METRICS
from trials import ROOT, TargetSet, load_trial, np, original


OUT = ROOT / "direction" / "ranking_results"
POLICIES = ("split_direct", "pooled_direct", "split_bernoulli")
FAMILIES = ("time", "treatment")


def sources(site):
    if site == "lanna":
        from multiperiod_trial import load_trial as load_lanna
        root = ROOT / "direction" / "multiperiod_results"
        return load_lanna, root / "benchmarks" / site, root / "fixed" / site
    if site in ("tcse", "ultuna"):
        root = ROOT / "direction"
        return load_trial, root / "results" / site, root / "joint_results" / site
    raise ValueError(f"Unknown trial {site}")


def objective_weights(target, domains_per_family, split):
    if not split:
        return target.primary_weights[target.primary][None, :]
    weights = np.zeros((len(FAMILIES), len(target.C)))
    for _, _, family, a, b, _ in target.groups:
        if family in FAMILIES:
            weights[FAMILIES.index(family), a:b] = 1. / (domains_per_family[family] * (b - a))
    return weights[:, target.primary]


def local_gains(state, sign_weights, bernoulli):
    ids = np.flatnonzero(state.available)
    den = state.R.diagonal()[ids]
    assert np.all(den > 0)
    # Sign and magnitude apply to the same contrast matrix, posterior and paid set.
    mse_weights = state.targets.mse_weights[state.targets.primary]
    magnitude = state.scale * (mse_weights @ (state.cross[:, ids] ** 2)) / den
    unresolved = state.unobserved > 0
    var = state.var[unresolved]
    assert np.all(var > 0)
    z = np.abs(state.mu[unresolved]) / np.sqrt(state.scale * var)
    r = np.abs(state.cross[np.ix_(unresolved, ids)]) / np.sqrt(var[:, None] * den)
    sign = sign_weights[:, unresolved] @ sign_gain(z[:, None], r, bernoulli)
    return state.channel.rows[ids], np.vstack([sign, magnitude])


def allocation(size, panels, channels, targets, ratio, pilot, oracle, policy, seed):
    # pooled_bernoulli is the archived joint rule; the verifier replays that cell independently.
    if policy not in (*POLICIES, "pooled_bernoulli"):
        raise KeyError(policy)
    split = policy.startswith("split_")
    bernoulli = policy.endswith("_bernoulli")
    counted = {family: sum(group[2] == family for t in targets for group in t.groups)
               for family in FAMILIES}
    assert min(counted.values()) > 0
    weights = [objective_weights(t, counted, split) for t in targets]
    paid = oracle.reveal(pilot)
    states = [PosteriorState(ch, t, ratio, pilot, paid[panels[ch.panel_ids[0]].name], "sign_error")
              for ch, t in zip(channels, targets)]
    lookup = [{int(row): i for i, row in enumerate(ch.rows)} for ch in channels]
    priority = np.empty(size, int)
    priority[np.random.default_rng(seed).permutation(size)] = np.arange(size)
    available = np.ones(size, dtype=bool)
    available[pilot] = False
    path = list(pilot)
    while available.any():
        gains = np.zeros((len(weights[0]) + 1, size))
        for state, w in zip(states, weights):
            rows, local = local_gains(state, w, bernoulli)
            gains[:, rows] += local
        choice = choose(np.flatnonzero(available), gains, priority)
        measured = oracle.reveal(np.array([choice]))
        for ch, local_index, state in zip(channels, lookup, states):
            if choice in local_index:
                state.update(local_index[choice], measured[panels[ch.panel_ids[0]].name][0])
        available[choice] = False
        path.append(choice)
    assert len(oracle.paid) == size
    return np.asarray(path, dtype=np.int16)


def run(site, start, stop, output=OUT):
    loader, baseline, fixed = sources(site)
    cfg = json.loads((baseline / "config.json").read_text(encoding="utf-8"))
    frame, panels, channels, responses, exact = loader(site)
    size = len(frame)
    targets = [TargetSet(panels, ch) for ch in channels]
    evaluator = Evaluator(panels, channels, targets, exact)
    coords = [[d, f] for t in targets for _, d, f, *_ in t.groups]
    assert cfg["coordinates"] == coords and cfg["models"] == MODELS and cfg["metrics"] == METRICS
    assert size == cfg["records"] and cfg["ratio"] == 1
    covs = [ch.kernel + cfg["ratio"] * np.eye(len(ch.rows)) for ch in channels]
    budgets = cfg["budgets"]
    folder = output / site
    folder.mkdir(parents=True, exist_ok=True)
    for rep in range(start, stop):
        dest = folder / f"draw_{rep:03d}.npz"
        if dest.exists():
            continue
        seed = cfg["seed"] + rep
        pilot = np.random.default_rng(seed).choice(size, budgets[0], replace=False)
        initial = {k: v[pilot] for k, v in responses.items()}
        settings = {p.name: original.pilot_settings(p, pilot, initial[p.name]) for p in panels}
        with np.load(baseline / dest.name) as reference:
            np.testing.assert_array_equal(reference["pilot"], pilot)
            pilot_score = reference["scores"][0, 0].copy()
        if rep == 0:
            known = allocation(size, panels, channels, targets, cfg["ratio"], pilot,
                               original.PaidOracle(responses), "pooled_bernoulli", seed + 100000)
            with np.load(fixed / dest.name) as saved:
                np.testing.assert_array_equal(saved["pilot"], pilot)
                np.testing.assert_array_equal(saved["paths"][0], known)
        sizes = (len(POLICIES), len(budgets), len(MODELS), len(coords), len(METRICS))
        scores = np.empty(sizes)
        paths = []
        for di, policy in enumerate(POLICIES):
            path = allocation(size, panels, channels, targets, cfg["ratio"], pilot,
                              original.PaidOracle(responses), policy, seed + 100000)
            np.testing.assert_array_equal(path[:len(pilot)], pilot)
            np.testing.assert_array_equal(np.sort(path), np.arange(size))
            for bi, budget in enumerate(budgets):
                scores[di, bi] = score_set(path[:budget], panels, channels, targets, covs,
                                           settings, responses, evaluator, size)
            paths.append(path)
            np.testing.assert_array_equal(scores[di, 0], pilot_score)
        assert np.isfinite(scores).all() and not np.any(scores[:, -1])
        temp = dest.with_suffix(".tmp")
        with temp.open("wb") as stream:
            np.savez_compressed(stream, scores=scores, paths=np.stack(paths),
                                pilot=pilot, policies=np.asarray(POLICIES))
        temp.replace(dest)
    return site, start, stop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", nargs="+", choices=["tcse", "ultuna", "lanna"],
                        default=["tcse", "ultuna", "lanna"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=200)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    if not 0 <= args.start < args.stop <= 200 or args.workers < 1:
        parser.error("Provide a nonempty subset of the 200 pilots and a positive worker count.")
    if args.workers == 1:
        for site in args.sites:
            run(site, args.start, args.stop, args.output_dir)
        return
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run, site, lo, min(lo + 5, args.stop), args.output_dir)
                for lo in range(args.start, args.stop, 5) for site in args.sites]
        for job in as_completed(jobs):
            print(*job.result(), flush=True)


if __name__ == "__main__":
    main()
