"""Typeset the Ultuna 50-extraction example from the archived replay summaries."""
import argparse
import csv
import math
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
RULES = ("joint_maximin", "contrast_mse", "sign_error", "alternate")
DOMAINS = ("P_soil_0_20_cm", "K_soil_0_20_cm")


def fixed_records(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            if (row["site"] == "ultuna" and row["model"] == "kriging"
                    and row["budget"] == "50" and row["design"] in RULES):
                yield row


def index_records(rows, fields):
    index = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in index:
            raise ValueError(f"Duplicate summary row for {key}")
        index[key] = row
    return index


def temporal_counts(inventory_path):
    with inventory_path.open(encoding="utf-8-sig", newline="") as source:
        inventory = [row for row in csv.DictReader(source) if row["family"] == "time"]
    counts = {}
    for domain in DOMAINS:
        rows = [row for row in inventory if row["domain"] == domain]
        magnitudes = [abs(Fraction(row["reference_exact"])) for row in rows]
        nonzero = sorted(value for value in magnitudes if value)
        if not nonzero:
            raise ValueError(f"No nonzero temporal comparisons for {domain}")
        cutoff = nonzero[(len(nonzero) - 1) // 4]
        small = [0 < value <= cutoff for value in magnitudes]
        if small != [row["small"] == "True" for row in rows] or sum(small) >= len(nonzero):
            raise ValueError(f"Invalid temporal quartile flags for {domain}")
        counts[domain] = (len(nonzero), sum(small))
    return counts


def fixed_budget_table(results):
    scores = index_records(fixed_records(results / "performance.csv"), ("design", "group", "metric"))
    domains = index_records(fixed_records(results / "domains.csv"), ("design", "domain", "family"))

    counts = temporal_counts(results.parent / "results" / "ultuna" / "contrast_inventory.csv")

    def signed_errors(rule, family):
        summary = float(scores[rule, family, "direction"]["mean"])
        by_element = [float(domains[rule, domain, family]["direction"]) for domain in DOMAINS]
        if not 0 <= summary <= 1 or not math.isclose(summary, sum(by_element) / len(by_element),
                                                     rel_tol=1e-12, abs_tol=1e-14):
            raise ValueError(f"Inconsistent {family} direction score for {rule}")
        return 100 * summary

    def above_quartile(rule):
        rates = []
        for domain in DOMAINS:
            cell = domains[rule, domain, "time"]
            total, small = counts[domain]
            wrong_all = total * float(cell["direction"])
            wrong_small = small * float(cell["small_direction"])
            rates.append((wrong_all - wrong_small) / (total - small))
        if any(rate < -1e-12 or rate > 1 + 1e-12 for rate in rates):
            raise ValueError(f"Invalid above-quartile sign error for {rule}")
        return 100 * sum(rates) / len(rates)

    def scaled_mse(rule):
        summary = float(scores[rule, "primary", "scaled_mse"]["mean"])
        cells = [float(domains[rule, domain, family]["scaled_mse"])
                 for domain in DOMAINS for family in ("time", "treatment")]
        if summary < 0 or not math.isclose(summary, sum(cells) / len(cells), rel_tol=1e-12):
            raise ValueError(f"Inconsistent combined MSE for {rule}")
        return summary

    def temporal_sign_error(rule, domain):
        value = float(domains[rule, domain, "time"]["direction"])
        if not 0 <= value <= 1:
            raise ValueError(f"Invalid temporal sign error for {rule}, {domain}")
        return 100 * value

    def temporal_rmse(rule, domain):
        value = float(domains[rule, domain, "time"]["native_mse"])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid temporal mean squared error for {rule}, {domain}")
        return math.sqrt(value)

    rows = [
        ("Temporal sign error (\\%)", [signed_errors(rule, "time") for rule in RULES], 2),
        ("Temporal sign error above first quartile (\\%)", [above_quartile(rule) for rule in RULES], 2),
        ("Treatment sign error (\\%)", [signed_errors(rule, "treatment") for rule in RULES], 2),
        ("Combined scaled MSE", [scaled_mse(rule) for rule in RULES], 3),
        ("Temporal P sign error (\\%)", [temporal_sign_error(rule, DOMAINS[0]) for rule in RULES], 2),
        ("Temporal P RMSE (mg P kg$^{-1}$)", [temporal_rmse(rule, DOMAINS[0]) for rule in RULES], 2),
        ("Temporal K sign error (\\%)", [temporal_sign_error(rule, DOMAINS[1]) for rule in RULES], 2),
        ("Temporal K RMSE (mg K kg$^{-1}$)", [temporal_rmse(rule, DOMAINS[1]) for rule in RULES], 2),
    ]

    lines = [
        r"\begin{table}[!htbp]\centering\small",
        r"\caption{Recovery at Ultuna with 50 of 100 plot--year specimens assayed, including the ten-specimen pilot. "
        r"Values use common kriging and average over 200 shared pilots. For alternation, the two starting orders "
        r"have equal weight within each pilot. The quartile cutoff is applied within each element to the absolute "
        r"nonzero temporal changes. Temporal root mean squared error (RMSE) is the square root of "
        r"the mean squared error for changes in hydrochloric-acid-extractable P or K between 1979 and 1989.}",
        r"\label{tab:ultuna-fixed-budget}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrr@{}}\toprule",
        r"Outcome & Joint & Magnitude & Direct sign & Alternation\\\midrule",
    ]
    for label, values, digits in rows:
        lines.append(" & ".join([label, *(f"{n:.{digits}f}" for n in values)]) + r"\\")
    return "\n".join([*lines, r"\bottomrule\end{tabular*}", r"\end{table}", ""])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE.parent / "Supplement" / "UltunaFixedBudgetTable.tex")
    args = parser.parse_args()
    table = fixed_budget_table(HERE / "joint_results")
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(table, encoding="utf-8", newline="\n")

