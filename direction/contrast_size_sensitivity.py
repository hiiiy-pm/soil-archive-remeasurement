"""Rescore the saved, unchanged selections after excluding the smallest reference contrasts."""
import argparse
import csv
import io
import json
from fractions import Fraction
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
COLLECTIONS = {
    "Lincoln-TCSE": ("results/tcse", "joint_results/tcse"),
    "Ultuna": ("results/ultuna", "joint_results/ultuna"),
    "Lanna": ("multiperiod_results/benchmarks/lanna", "multiperiod_results/fixed/lanna"),
    "Högåsa": ("external_results/benchmarks/hogasa", "external_results/fixed/hogasa"),
    "Park Grass": ("alert_results/benchmarks/parkgrass", "alert_results/fixed/parkgrass"),
    "Hailun": ("olsen_results/benchmarks/hailun", "olsen_results/fixed/hailun"),
    "Winchmore": ("pilot_olsen_results/fixed_designs/benchmarks/winchmore",
                  "pilot_olsen_results/fixed_designs/fixed/winchmore"),
}
COMPARATORS = {"time": "contrast_mse", "treatment": "sign_error"}


def contrast_counts(directory, coordinates):
    with (directory / "contrast_inventory.csv").open(encoding="utf-8", newline="") as stream:
        inventory = list(csv.DictReader(stream))
    totals, smallest = [], []
    for domain, family in coordinates:
        group = [row for row in inventory if (row["domain"], row["family"]) == (domain, family)]
        magnitudes = [abs(Fraction(row["reference_exact"])) for row in group]
        nonzero = sorted(value for value in magnitudes if value)
        cutoff = nonzero[(len(nonzero) - 1) // 4]
        small = [0 < value <= cutoff for value in magnitudes]
        if small != [row["small"] == "True" for row in group]:
            raise ValueError(f"Archived quartile flags disagree for {domain}, {family}")
        totals.append(len(nonzero))
        smallest.append(sum(small))
    return np.array(totals), np.array(smallest)


def read_scores(archive, member, strategies, model, metrics):
    with np.load(io.BytesIO(archive.read(member))) as replay:
        scores = replay["scores"][strategies, :-1, model]
    return scores[..., metrics].mean(axis=1)


def archive_results(site, baseline, joint):
    config = json.loads((baseline / "config.json").read_text(encoding="utf-8"))
    coordinates = config["coordinates"]
    total, small = contrast_counts(baseline, coordinates)
    model = config["models"].index("kriging")
    metrics = [config["metrics"].index(name) for name in ("direction", "small_direction")]
    strategies = [config["designs"].index(rule) for rule in COMPARATORS.values()]
    scores = np.zeros((len(strategies) + 1, len(coordinates), len(metrics)))
    with ZipFile(baseline / "selections.zip") as base_draws, ZipFile(joint / "selections.zip") as joint_draws:
        for draw in range(config["replicates"]):
            member = f"draw_{draw:03d}.npz"
            scores[:-1] += read_scores(base_draws, member, strategies, model, metrics)
            scores[-1:] += read_scores(joint_draws, member, [0], model, metrics)
    scores /= config["replicates"]
    for index, (family, comparator) in enumerate(COMPARATORS.items()):
        use = [i for i, (_, name) in enumerate(coordinates) if name == family]
        n_all, n_small = total[use], small[use]
        if np.any(n_all <= n_small):
            raise ValueError(f"No contrasts above the cutoff in a contributing {site} domain")
        # Each stored error rate is the fraction wrong in that domain and family.
        # Subtract the errors in the smallest group before taking the equal-domain mean.
        rates = scores[[index, -1]][:, use]
        full = rates[:, :, 0]
        above = (n_all * full - n_small * rates[:, :, 1]) / (n_all - n_small)
        yield {
            "archive": site, "family": family, "comparator": comparator,
            "n_nonzero": int(n_all.sum()), "n_above_cutoff": int((n_all - n_small).sum()),
            "all_comparator_pct": 100 * full[0].mean(), "all_joint_pct": 100 * full[1].mean(),
            "above_cutoff_comparator_pct": 100 * above[0].mean(),
            "above_cutoff_joint_pct": 100 * above[1].mean(),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write site-level sign error percentages to this CSV")
    args = parser.parse_args()
    rows = [row for site, (baseline, joint) in COLLECTIONS.items()
            for row in archive_results(site, HERE / baseline, HERE / joint)]
    result = pd.DataFrame(rows)
    if args.output:
        result.to_csv(args.output, index=False, float_format="%.8f")
    else:
        print(result.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
