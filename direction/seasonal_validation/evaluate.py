"""Replay the five prespecified quota-constrained designs on the frozen external frame."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "harvard_forest"))
from replay import PATH_NAMES, REPORT_NAMES, original
from trials import Channel, TargetSet, discrepancy_kernel
from horizon_balance import constrained_path
from explain_replays import average_orders, score_path
import numpy as np
import pandas as pd
from scipy.stats import t


def schedule(strata, pilot, seed):
    """Largest proportional deficit after the next purchase, shared ties and pilot."""
    size = np.bincount(strata)
    totals = np.bincount(strata[pilot], minlength=len(size))
    out = []
    rng = np.random.default_rng(seed + 300000)
    for step in range(len(pilot) + 1, len(strata) + 1):
        deficit = step * size - len(strata) * totals
        allowed = np.flatnonzero(totals < size)
        tied = allowed[deficit[allowed] == deficit[allowed].max()]
        choice = int(tied[0] if len(tied) == 1 else rng.choice(tied))
        totals[choice] += 1
        out.append(choice)
    return np.array(out)


def run(output):
    lock = json.loads((HERE / "lock.json").read_text("utf-8-sig"))
    for name, digest in lock["files"].items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == digest, name
    frame = pd.read_csv(HERE / "frame.csv", dtype={"sample":str,"stratum":str,"group":str,"plot":str})
    assay = pd.read_csv(HERE / "references.csv", dtype={"sample":str})
    assert frame["sample"].is_unique and assay["sample"].is_unique
    y = assay.set_index("sample").loc[frame["sample"], "reference"].to_numpy()
    x = frame.auxiliary.to_numpy()
    df = pd.DataFrame(dict(year=frame.campaign, treatment=frame.group, depth=frame.stratum,
                           plot=frame["plot"], record_id=frame["sample"]))
    labels, strata = np.unique(frame.stratum, return_inverse=True)
    panels = [original.engine.make_panel(h, df, x, strata == k, lock["unit"], len(labels))
              for k, h in enumerate(labels)]
    channel = Channel("C", np.arange(len(x)), x, discrepancy_kernel(df, x), list(range(len(labels))))
    target = TargetSet(panels, channel)
    inventory = pd.read_csv(HERE / "limits.csv", dtype={"stratum":str,"group":str})
    selectors = []
    lookup = {}
    for pi, _, family, start, stop, _ in target.groups:
        if family != "time":
            continue
        for i in range(start, stop):
            _, cell, _ = target.definitions[i]
            (late, group), (early, group2) = [panels[pi].cells[j] for j in cell]
            assert group == group2
            lookup[(labels[pi], group, early, late)] = i
    for row in inventory.itertuples():
        selectors.append(lookup[(row.stratum,row.group,row.early,row.late)])
    assert set(selectors) == set(lookup.values()) and len(selectors) == len(lookup)
    C = target.C[selectors]
    actual = C @ y
    inventory["measured_change"] = actual
    inventory["different_seasons"] = (inventory.late % 10) != (inventory.early % 10)
    inventory["above_limit"] = abs(actual) > inventory.limit
    inventory["eligible"] = inventory.different_seasons & inventory.above_limit
    n = len(x)
    budgets = np.array([(n * j + 5) // 10 for j in range(1, 11)])
    assert len(set(budgets)) == 10 and budgets[0] >= 2
    seeds = np.arange(736400, 736600)
    paths = np.empty((len(seeds), len(PATH_NAMES), n), dtype=np.int16)
    pred = np.empty(paths.shape[:2] + (len(budgets), len(actual)))
    mse = np.empty(pred.shape[:3])
    for j, seed in enumerate(seeds):
        pilot = np.random.default_rng(seed).choice(n, budgets[0], replace=False)
        horizons = schedule(strata, pilot, seed)
        for k, policy in enumerate(PATH_NAMES):
            oracle = original.PaidOracle({"C": y})
            path = constrained_path(strata, pilot, horizons, channel, target, policy, seed, oracle)
            np.testing.assert_array_equal(strata[path[len(pilot):]], horizons)
            np.testing.assert_array_equal(np.sort(path), np.arange(n))
            paths[j, k] = path
            pred[j, k], mse[j, k] = score_path(path, budgets, channel, target, y, C, actual)
    assert np.all(pred[:, :, 0] == pred[:, :1, 0]) and np.all(pred[:, :, -1] == actual)
    assert np.all(mse[:, :, 0] == mse[:, :1, 0]) and np.all(mse[:, :, -1] == 0)
    output.mkdir(parents=True, exist_ok=True)
    errors = average_orders(100. * (np.sign(pred) != np.sign(actual)))
    gates = inventory.eligible.to_numpy()
    layer_metrics = {}
    for label in labels:
        mask = gates & (inventory.stratum == label).to_numpy()
        if mask.any():
            layer_metrics["sign_" + str(label)] = errors[..., mask].mean(axis=-1)
    metrics = {"scaled_mse": average_orders(mse)}
    if layer_metrics:
        metrics["sign_equal_strata"] = np.mean(list(layer_metrics.values()), axis=0)
        metrics["sign_equal_contrasts"] = errors[..., gates].mean(axis=-1)
    metrics.update(layer_metrics)
    records = []
    for metric, values in metrics.items():
        settings = [(str(b), values[:, :, i]) for i,b in enumerate(budgets)]
        settings.append(("partial_mean", values[:, :, :-1].mean(axis=2)))
        for budget, draws in settings:
            for k,name in enumerate(REPORT_NAMES):
                delta = draws[:, 0] - draws[:, k]
                sem = delta.std(ddof=1)/np.sqrt(len(seeds))
                mu = delta.mean()
                half95 = t.ppf(.975,len(seeds)-1)*sem
                half975 = t.ppf(.9875,len(seeds)-1)*sem
                records.append((metric,budget,name,draws[:, k].mean(),mu,mu-half95,mu+half95,mu-half975,mu+half975))
    for k,name in enumerate(REPORT_NAMES):
        inventory[name+"_wrong_pct"] = errors[:, k, :-1].mean(axis=(0,1))
    inventory.to_csv(output/"contrasts.csv",index=False)
    pd.DataFrame(records,columns=("metric","budget","policy","mean","joint_minus_policy","low95","high95",
                                 "low97_5","high97_5")).to_csv(output/"summary.csv",index=False)
    np.savez_compressed(output/"replays.npz",paths=paths,predictions=pred,scaled_mse=mse,
                        budgets=budgets,seeds=seeds,policies=np.array(PATH_NAMES),samples=np.array(frame["sample"],dtype=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    run(parser.parse_args().output)

