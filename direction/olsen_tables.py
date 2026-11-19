"""Typeset the equal-budget Olsen P replay from its saved numerical summaries."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from joint_tables import NAMES


HERE = Path(__file__).resolve().parent
SITES = {"parkgrass": "Park Grass", "hailun": "Hailun"}
LABELS = {
    **NAMES, "joint_maximin": "Joint selection", "gap_joint": "Olsen-weighted joint",
    "gap_only": "Olsen-only allocation", "pilot_warning": "Pilot warning",
    "split_direct": "Separate sign risks", "pooled_direct": "Pooled sign risk",
    "split_bernoulli": "Separate sign variances",
}
POLICIES = ["gap_joint", "joint_maximin", "gap_only", "pilot_warning", "alternate",
            "split_direct", "pooled_direct", "split_bernoulli",
            *[key for key in NAMES if key not in {"joint_maximin", "alternate"}]]
GROUPS = {"treatment": "Treatment", "time": "Time", "primary": "Magnitude"}
DEPTHS = {"P_topsoil": "0--23", "P_0-10": "0--10", "P_0-20": "0--20"}
FAMILIES = {"treatment": "Treatment", "time": "Time", "interaction": "Interaction"}
BREAK = r"\\"


def start_table(label, caption, columns, header):
    return [r"\begin{table}[!htbp]\centering\footnotesize",
            r"\caption{" + caption + r"}\label{tab:" + label + "}",
            r"\setlength{\tabcolsep}{3pt}\begin{tabular}{@{}" + columns + r"@{}}\toprule", header]


def finish_table(lines):
    return "\n".join([*lines, r"\bottomrule\end{tabular}\end{table}"])


def primary(tab, model="kriging"):
    result = tab.loc[(tab["model"] == model) & (tab["period"] == "noncensus")]
    assert (result.n == 200).all()
    return result


def value(x, group, delta=False):
    scaled = x * (1 if group == "primary" else 100)
    digits = 3 if group == "primary" or delta else 2
    while scaled != 0 and round(scaled, digits) == 0 and digits < 8:
        digits += 1
    return f"{scaled:.{digits}f}"


def means_table(performance):
    lines = start_table(
        "olsen-all",
        r"Equal-budget comparison of all 26 allocations. Treatment and temporal direction errors "
        r"are percentages. Magnitude is the combined scaled mean squared error (MSE). Values average "
        r"the nine non-census budgets within each of 200 matched pilots using common kriging. "
        r"Bold type identifies the lowest observed loss in each column, including ties.",
        "lrrrrrr", r"&\multicolumn{3}{c}{Park Grass}&\multicolumn{3}{c}{Hailun}\\"
        + "\n" + r"\cmidrule(lr){2-4}\cmidrule(l){5-7}"
        + "\n" + r"Allocation & Treatment (\%) & Time (\%) & MSE & Treatment (\%) & Time (\%) & MSE\\\midrule")
    columns = {}
    for site, tab in performance.items():
        columns[site] = primary(tab).pivot(index="design", columns="group", values="mean")
        assert set(columns[site].index) == set(POLICIES)
    for rule in POLICIES:
        parts = [LABELS[rule]]
        if rule == "sign_error":
            lines.append(r"\midrule")
        for tab in columns.values():
            for group in GROUPS:
                loss = tab.at[rule, group]
                printed = value(loss, group)
                if np.isclose(loss, tab[group].min(), rtol=1e-13, atol=1e-14):
                    printed = r"\textbf{" + printed + "}"
                parts.append(printed)
        lines.append(" & ".join(parts) + BREAK)
    return finish_table(lines)


def differences_table(root):
    lines = start_table(
        "olsen-paired",
        r"Paired losses for the Olsen-weighted rule minus each specified comparator. Negative values "
        r"favour Olsen weighting. Differences in direction error use percentage points. "
        r"Magnitude uses scaled MSE. Brackets contain simultaneous 95\% Monte Carlo intervals "
        r"over the two sites, three outcomes and 25 comparators. The lowest-loss comparator is "
        r"identified separately for each site and outcome, including ties.",
        "llllr", r"Site & Outcome & Comparison & Allocation & Difference [interval]\\\midrule")
    for site, name in SITES.items():
        paired = primary(pd.read_csv(root/site/"paired_differences.csv")).set_index(["group", "design"])
        best = pd.read_csv(root/site/"strongest.csv")
        for g, group in enumerate(GROUPS):
            component = {"treatment": "joint_maximin", "time": "contrast_mse", "primary": "sign_error"}[group]
            roles = {"joint_maximin": ["Joint"]}
            if component != "joint_maximin":
                roles[component] = ["Component"]
            for winner in best.loc[best["group"] == group, "design"]:
                roles.setdefault(winner, []).append("Best")
            for i, (rule, role) in enumerate(roles.items()):
                row = paired.loc[(group, rule)]
                assert row.lower <= row["mean"] <= row.upper
                m, lo, hi = (value(row[k], group, delta=True) for k in ["mean", "lower", "upper"])
                interval = "$" + m + r"\;[" + lo + "," + hi + "]$"
                lines.append(" & ".join([name if i == g == 0 else "",
                                          GROUPS[group] if i == 0 else "", "/".join(role),
                                          LABELS[rule], interval]) + BREAK)
            lines.append(r"\addlinespace")
        if site != list(SITES)[-1]:
            lines.append(r"\midrule")
    return finish_table(lines)


def domains_table(root):
    lines = start_table(
        "olsen-domains",
        r"Direction and magnitude errors by depth and contrast family for the unweighted and "
        r"Olsen-weighted joint allocations under common kriging. Direction includes every nonzero "
        r"reference difference. Small denotes the lowest quartile of nonzero absolute reference "
        r"differences within each domain and family, with ties at the cutoff retained. Three-class "
        r"error also scores exact equality. RMSE is the square root of the mean native-unit squared "
        r"error over pilots and non-census budgets, in mg P kg$^{-1}$. Hailun's recorded 0--10 cm "
        r"domain contains within-year treatment comparisons in 2004 alone.",
        "lllrrrr", r"Depth (cm) & Family & Allocation & Direction (\%) & Small (\%) & Three-class (\%) & RMSE\\\midrule")
    for site, label in SITES.items():
        cfg = json.loads((root/site/"config.json").read_text(encoding="utf-8"))
        domain = pd.read_csv(root/site/"domain_results.csv")
        frame = domain.loc[domain.model == "kriging"].set_index(["domain", "family", "design", "metric"])
        lines.append(r"\multicolumn{7}{@{}l}{\emph{" + label + r"}}" + BREAK)
        for depth, family in cfg["coordinates"]:
            for rule in ["joint_maximin", "gap_joint"]:
                cells = [DEPTHS[depth] if rule == "joint_maximin" else "",
                         FAMILIES[family] if rule == "joint_maximin" else "",
                         "Unweighted" if rule == "joint_maximin" else "Olsen-weighted"]
                for metric in ["direction", "small_direction", "three_class", "native_mse"]:
                    row = frame.loc[(depth, family, rule, metric)]
                    assert row.n == 200 and row["mean"] >= 0
                    if metric == "native_mse":
                        cells.append(f"{np.sqrt(row['mean']):.2f}")
                    else:
                        cells.append(value(row["mean"], "direction"))
                lines.append(" & ".join(cells) + BREAK)
            lines.append(r"\addlinespace")
        if site != list(SITES)[-1]:
            lines.append(r"\midrule")
    return finish_table(lines)


def sensitivity_table(performance):
    lines = start_table(
        "olsen-ridge",
        r"Ridge reconstruction sensitivity at identical budget levels. The allocation paths are "
        r"held fixed, and every acquired reference value is retained in reconstruction. Direction "
        r"errors are percentages and MSE is the combined scaled loss. All entries average the nine "
        r"non-census budgets within each of 200 matched pilots.",
        "llrrr", r"Site & Allocation & Treatment (\%) & Time (\%) & MSE\\\midrule")
    for site, label in SITES.items():
        tab = primary(performance[site], "ridge").pivot(index="design", columns="group", values="mean")
        for i, rule in enumerate(["gap_joint", "joint_maximin", "gap_only", "sign_error", "contrast_mse"]):
            parts = [label if i == 0 else "", LABELS[rule]]
            parts.extend(value(tab.at[rule, g], g) for g in GROUPS)
            lines.append(" & ".join(parts) + BREAK)
        if site != list(SITES)[-1]:
            lines.append(r"\midrule")
    return finish_table(lines)


def render(root, output):
    performance = {site: pd.read_csv(root/site/"performance.csv") for site in SITES}
    tables = [means_table(performance), differences_table(root), domains_table(root),
              sensitivity_table(performance)]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n\n".join(tables) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE/"olsen_results")
    parser.add_argument("--output", type=Path,
                        default=HERE.parents[1]/"typeset"/"source"/"OlsenTables.tex")
    args = parser.parse_args()
    render(args.results, args.output)
