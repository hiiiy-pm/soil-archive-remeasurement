"""Describe the treatment contrast with the highest mean pilot warning probability.

The example was specified before examining individual alert probabilities or
observed error rates. Select on mean frozen q over all 200 pilots, including a
zero for a fully resolved contrast, then on the lowest source contrast index.
The reference difference and realised errors are used only after selection.
"""
import argparse
import csv
import json
from fractions import Fraction
from pathlib import Path

from alert_trial import load_trial
import numpy as np


OUT = Path(__file__).resolve().parent / "alert_results"


def predict_reference(channel, covariance, paid, values):
    """Fit the ordinary-kriging discrepancy with an unknown intercept."""
    size = len(paid)
    system = np.ones((size + 1, size + 1))
    system[:-1, :-1] = covariance[np.ix_(paid, paid)]
    system[-1, -1] = 0.
    solution = np.linalg.solve(system, np.r_[values - channel.x[paid], 0.])
    cross = np.column_stack([covariance[:, paid], np.ones(len(covariance))])
    prediction = channel.x + cross @ solution
    prediction[paid] = values
    return prediction


def describe(output):
    base = output / "benchmarks" / "parkgrass"
    cfg = json.loads((base / "config.json").read_text())
    files = [output/"warning"/f"draw_{i:03d}.npz" for i in range(200)]
    risk = []
    for file in files:
        with np.load(file) as saved:
            values = saved["risk"]
            assert np.all(values[~saved["initially_unresolved"]] == 0.)
            risk.append(values)
    mean_risk = np.mean(risk,axis=0)
    chosen = int(np.argmax(mean_risk))

    frame, [panel], [ch], observed, exact = load_trial()
    ids, coefficients = panel.K["treatment"]
    assert len(ids) == len(mean_risk)
    a, b = ids[chosen]
    np.testing.assert_array_equal(coefficients[chosen], [1,-1])
    (year, first), (second_year, second) = [panel.cells[j] for j in (a,b)]
    assert year == second_year
    first_set = panel.rows[panel.g[panel.rows] == a]
    second_set = panel.rows[panel.g[panel.rows] == b]
    weights = np.zeros(len(frame))
    weights[first_set] = 1/len(first_set)
    weights[second_set] = -1/len(second_set)
    reference = exact[panel.name]
    lab_mean = lambda group: sum((reference[i] for i in group),Fraction()) / len(group)
    left, right = lab_mean(first_set), lab_mean(second_set)
    true_difference = left-right
    true_sign = int(true_difference>0)-int(true_difference<0)
    values = observed[panel.name]
    covariance = ch.kernel + cfg["ratio"] * np.eye(len(frame))
    original_errors, warning_errors = [], []
    support = np.flatnonzero(weights)
    for file in files:
        with np.load(file) as saved:
            path = saved["paths"][-1]
            original_errors.append(saved["joint_wrong"][:,chosen])
            errors = []
            for budget in cfg["budgets"]:
                paid = path[:budget]
                if np.isin(support, paid).all():
                    errors.append(False)
                    continue
                predicted = predict_reference(ch,covariance,paid,values[paid])
                errors.append(np.sign(weights @ predicted) != true_sign)
            warning_errors.append(errors)
    original_errors, warning_errors = map(np.array,(original_errors,warning_errors))
    assert not (original_errors[:,-1].any() or warning_errors[:,-1].any())
    rows = []
    for budget, subset in [("noncensus",slice(0,-1))] + list(zip(cfg["budgets"],range(len(cfg["budgets"])))):
        observed_joint = original_errors[:,subset]
        observed_warning = warning_errors[:,subset]
        rows.append(dict(
            period=budget, treatment_contrast_index=chosen+1, year=year, first=first, second=second,
            first_n=len(first_set), second_n=len(second_set), first_reference_exact_mg_kg=str(left),
            second_reference_exact_mg_kg=str(right), difference_exact_mg_kg=str(true_difference),
            difference_mg_kg=float(true_difference), mean_frozen_risk=mean_risk[chosen],
            pilot_budget_settings=observed_joint.size, joint_wrong=int(observed_joint.sum()),
            warning_wrong=int(observed_warning.sum()), joint_error_rate=observed_joint.mean(),
            warning_error_rate=observed_warning.mean()
        ))
    with (output/"field_example.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(rows[0])


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    describe(parser.parse_args().output_dir)
