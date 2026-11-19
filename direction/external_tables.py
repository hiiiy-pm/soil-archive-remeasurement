"""Build the Högåsa supplementary tables from the archived trial summaries."""
import argparse
from pathlib import Path

import pandas as pd

from joint_tables import NAMES


HERE = Path(__file__).resolve().parent
ROOT = HERE / "external_results"
LABELS = {"gate": "Pilot choice", **NAMES}
ENDPOINTS = [("time", "direction"), ("treatment", "direction"), ("primary", "scaled_mse")]
FIXED = ["joint_maximin", "alternate"]
SITE = r"H\"og{\aa}sa"
ROW = r"\\"
END = r"\bottomrule\end{tabular}\end{table}"


def read_primary(folder, filename):
    rows = pd.read_csv(folder / filename, dtype={"budget": str})
    return rows[(rows.budget == "noncensus") & (rows.model == "kriging") & (rows.site == "hogasa")]


def table_all(absolute):
    s = [
        r"\begin{table}[htbp]\centering\small",
        r"\caption{Laboratory comparison errors at " + SITE +
        r" for every allocation under a common kriging predictor. Time and treatment sign errors "
        r"refer to every nonzero reference difference. Scaled MSE combines both families. "
        r"Three-class error includes reference ties. Small-contrast error includes the smallest "
        r"quarter of nonzero reference differences within each supported domain and family. "
        r"All error rates are percentages. Entries average nine non-census budgets within each "
        r"of 200 pilots and then average over pilots.}\label{tab:pilot-all}",
        r"\setlength{\tabcolsep}{3pt}\begin{tabular}{@{}lrrrrr@{}}\toprule",
        r" & \multicolumn{2}{c}{Sign error (\%)} & \multicolumn{3}{c}{Treatment and time combined}\\",
        r"\cmidrule(lr){2-3}\cmidrule(l){4-6}",
        r"Allocation & Time & Treatment & Scaled MSE & Three-class (\%) & Small (\%)\\\midrule",
    ]
    if set(absolute.index.get_level_values("design")) != set(LABELS) or not absolute.index.is_unique:
        raise ValueError("Expected one set of losses per rule for all 21 rules.")
    for rule, label in LABELS.items():
        if rule == "sign_error":
            s.append(r"\midrule")
        values = [absolute.loc[(g, m, rule)] for g, m in
                  [*ENDPOINTS, ("primary", "three_class"), ("primary", "small_direction")]]
        formatted = [f"{v:.4f}" if i == 2 else f"{100*v:.2f}" for i, v in enumerate(values)]
        s.append(" & ".join([label, *formatted]) + ROW)
    return "\n".join(s + [END])


def table_pairs(pairs, winners):
    s = [
        r"\begin{table}[htbp]\centering\small",
        r"\caption{Pilot choice minus the fixed joint and alternating allocations and the strongest "
        r"of the other 18 benchmarks on each outcome at " + SITE +
        r". Intervals are paired 95\% Monte Carlo intervals adjusted simultaneously for "
        r"all 60 comparisons. Direction differences use percentage points. Negative differences "
        r"favor pilot choice. Bold comparator entries mark the lowest mean of the 18 benchmarks "
        r"for that column.}\label{tab:pilot-paired}",
        r"\setlength{\tabcolsep}{4pt}\begin{tabular}{@{}lrrr@{}}\toprule",
        r"Comparator & Time sign & Treatment sign & Scaled MSE\\\midrule",
    ]
    if pairs.simultaneous_lower.notna().sum() != 60:
        raise ValueError("The simultaneous family requires all 60 paired intervals.")
    compared = pairs.set_index(["comparator", "group", "metric"])
    strongest = list(dict.fromkeys(winners))
    for rule in [*FIXED, *strongest]:
        if rule == strongest[0]:
            s.append(r"\midrule")
        numbers, intervals = [], []
        for j, key in enumerate(ENDPOINTS):
            row = compared.loc[(rule, *key)]
            digits, scale = (2, 100) if j < 2 else (4, 1)
            lo, val, hi = [row[k] * scale for k in ["simultaneous_lower", "difference", "simultaneous_upper"]]
            assert lo <= val <= hi
            number = f"{val:.{digits}f}"
            numbers.append("$" + (r"\mathbf{" + number + "}" if winners[j] == rule else number) + "$")
            interval_digits = 3 if j < 2 else digits
            intervals.append(f"$[{lo:.{interval_digits}f}, {hi:.{interval_digits}f}]$")
        s.extend([" & ".join([LABELS[rule], *numbers]) + ROW,
                  " & " + " & ".join(intervals) + ROW + "[3pt]"])
    return "\n".join(s + [END])


def table_decision(quality, choices):
    n = int(len(choices))
    n_joint = int((choices.choice == FIXED[0]).sum())
    if n != 200 or choices.repeat.nunique() != n or not choices.choice.isin(FIXED).all():
        raise ValueError("The disposition requires all 200 pre-assay decisions.")
    s = [
        r"\begin{table}[htbp]\centering\small",
        r"\caption{Realized nine-budget loss of the pilot-chosen rule relative to the unchosen "
        r"rule within each pilot. Of the " + str(n) + " pilots, " + str(n_joint) +
        " selected joint allocation and " + str(n - n_joint) + r" selected alternation. "
        r"The hindsight minimum chooses the lower realized loss separately for each pilot and "
        r"outcome using the full archive. Sign losses are percentages, and their differences are "
        r"in percentage points. Magnitude uses combined scaled MSE.}\label{tab:pilot-choice}",
        r"\setlength{\tabcolsep}{4pt}\begin{tabular}{@{}lrrrrrr@{}}\toprule",
        r"&\multicolumn{3}{c}{Pilots by chosen loss}&\multicolumn{3}{c}{Mean loss}\\",
        r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
        r"Outcome & Lower & Tied & Higher & Pilot choice & Hindsight minimum & Difference\\\midrule",
    ]
    indexed = quality.set_index(["group", "metric"])
    for j,(group, metric) in enumerate(ENDPOINTS):
        row = indexed.loc[(group,metric)]
        counts = [int(row[c]) for c in ["chosen_strictly_better", "chosen_tied", "chosen_worse"]]
        assert sum(counts) == int(row.total) == n
        digits, scale = (2, 100) if j < 2 else (4, 1)
        values = [f"{row[f]*scale:.{digits}f}" for f in ("gate_mean","oracle_mean","oracle_gap")]
        s.append(" & ".join([["Temporal sign", "Treatment sign", "Magnitude"][j], *map(str,counts), *values]) + ROW)
    return "\n".join(s + [END])


def table_soil(domains, inventory, lowest_mse_rule):
    s = [
        r"\begin{table}[htbp]\centering\small",
        r"\caption{Contrasts in HCl-extractable P or K at " + SITE +
        r" by depth and comparison family. Counts use the complete reference archive at published "
        r"precision. Root MSE uses the numerical scale of the deposited HCl results. Squared errors are averaged "
        r"over the nine non-census budgets and all 200 pilots. The final column uses " +
        LABELS[lowest_mse_rule] + r", the benchmark with the lowest combined scaled MSE.}"
        r"\label{tab:pilot-nutrients}",
        r"\setlength{\tabcolsep}{3pt}\begin{tabular}{@{}llrrrrrr@{}}\toprule",
        r"&&\multicolumn{2}{c}{Contrasts}&\multicolumn{4}{c}{Root MSE (source units)}\\",
        r"\cmidrule(lr){3-4}\cmidrule(l){5-8}",
        r"Pool, depth & Family & Nonzero & Tied & Pilot choice & Joint & Alternating & Benchmark\\\midrule",
    ]
    inventory = inventory[inventory.family.isin(["time","treatment"])]
    grouped = inventory.groupby(["domain","family"],sort=True)
    look = domains.set_index(["domain","family","design"]).native_rmse
    for (domain,family), ref in grouped:
        element, soil, low, high, cm = domain.split("_")
        if soil != "soil" or cm != "cm":
            raise ValueError("Unrecognized depth label " + domain)
        n = int((ref.true_sign != 0).sum())
        values=[f"{look.loc[(domain,family,k)]:.2f}" for k in ["gate",*FIXED,lowest_mse_rule]]
        s.append(" & ".join([f"{element}, {low}--{high} cm",family.title(),str(n),str(len(ref)-n),*values])+ROW)
    return "\n".join(s + [END])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("ExternalTables.tex"))
    parser.add_argument("--results-root",type=Path,default=ROOT)
    args = parser.parse_args()
    folder = args.results_root / "hogasa"
    observed = read_primary(folder,"performance.csv").set_index(["group","metric","design"])["mean"]
    best = read_primary(folder,"best_baselines.csv").set_index(["group","metric"])
    winners = [best.loc[e,"best_original"] for e in ENDPOINTS]
    inventory = pd.read_csv(args.results_root/"benchmarks/hogasa/contrast_inventory.csv")
    tables = [
        table_all(observed),table_pairs(read_primary(folder,"paired.csv"),winners),
        table_decision(read_primary(folder,"decision_quality.csv"),pd.read_csv(folder/"pilot_choices.csv")),
        table_soil(read_primary(folder,"domains.csv"),inventory,winners[2])
    ]
    args.output.parent.mkdir(exist_ok=True,parents=True)
    args.output.write_text("\n\n".join(tables)+"\n",encoding="utf-8")
