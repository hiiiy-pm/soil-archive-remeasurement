"""Generate supplementary LaTeX tables from the full joint-policy summaries."""
import argparse
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
NAMES = {
    "joint_maximin": "Joint maximin gain",
    "alternate": "Alternating criteria",
    "sign_error": "Direct sign error",
    "sign_variance": "Bernoulli sign variance",
    "contrast_mse": "Matched contrast MSE",
    "all_contrast_mse": "All-family contrast MSE",
    "conversion_mse": "Kriging conversion",
    "concentration_mse": "Kriging concentration",
    "random": "Random continuation",
    "cell_balance": "Year--treatment balance",
    "target_cell_balance": "Target-cell balance",
    "maximin": "Historical maximin distance",
    "active_sampling": "Active sampling",
    "retained_contrast": "Ridge contrast priority",
    "conversion": "Ridge conversion priority",
    "concentration": "Ridge concentration priority",
    "a2_if_year": "Two-wave influence/year",
    "a4_loo_year": "Four-wave residual/year",
    "a4_loo_cell": "Four-wave residual/cell",
    "a4_loo_targetcell": "Four-wave residual/target cell",
}
SITES = {"tcse": "Lincoln-TCSE", "ultuna": "Ultuna"}
ENDPOINTS = [("time", "direction"), ("treatment", "direction"), ("primary", "scaled_mse")]


def primary_frame(name):
    frame = pd.read_csv(HERE / "joint_results" / (name + ".csv"), dtype={"budget": str})
    return frame[(frame.model == "kriging") & (frame.budget == "noncensus")]


def table_means(performance):
    values = performance.set_index(["site", "design", "group", "metric"])["mean"]
    assert set(performance.design) == set(NAMES)
    lines = [
        r"\begin{table}[htbp]\centering\small",
        r"\caption{Time sign error, treatment sign error and scaled mean squared error (MSE) under identical assay budgets. Sign errors are percentages within the indicated family, averaged across contributing soil depths or elements. MSE combines treatment and temporal differences with the prespecified domain and family weights. Entries average the nine non-census budgets and 200 paired pilots under common kriging.}\label{tab:joint-all}",
        r"\setlength{\tabcolsep}{2.5pt}\begin{tabular}{@{}lrrrrrr@{}}\toprule",
        r"&\multicolumn{3}{c}{Lincoln-TCSE}&\multicolumn{3}{c}{Ultuna}\\",
        r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
        r"Selection rule &Time (\%)&Treatment (\%)&MSE&Time (\%)&Treatment (\%)&MSE\\\midrule",
    ]
    for design, label in NAMES.items():
        if design == "sign_error":
            lines.append(r"\midrule")
        row = [label]
        for site in SITES:
            for group, metric in ENDPOINTS:
                value = values.loc[(site, design, group, metric)]
                if metric == "direction":
                    row.append(f"{100 * value:.2f}")
                else:
                    row.append(f"{value:.5f}" if site == "tcse" else f"{value:.3f}")
        lines.append(" & ".join(row) + r"\\")
    return "\n".join(lines + [r"\bottomrule\end{tabular}\end{table}"])


def table_differences(paired, best, site):
    rows = paired[(paired.focal == "joint_maximin") & (paired.site == site)]
    diff = rows.set_index(["comparator", "group", "metric"])
    winners = best[(best.site == site) & (best.focal == "joint_maximin")].set_index(["group", "metric"])
    original = [winners.loc[e, "best_original"] for e in ENDPOINTS]
    comparators = list(dict.fromkeys(["sign_error", "contrast_mse", "alternate", "sign_variance", *original]))
    lines = [
        r"\begin{table}[htbp]\centering\small",
        r"\caption{Joint maximin gain minus competing allocations at " + SITES[site] +
        r". Each estimate is followed by its simultaneous 95\% Monte Carlo interval on the next line. Sign differences are percentage points and MSE differences use the scaled units of Table~\ref{tab:joint-all}. The intervals account for 114 comparisons across both sites, all 19 competitors and the three endpoints. Negative differences favour joint maximin gain.}\label{tab:joint-diff-" + site + "}",
        r"\setlength{\tabcolsep}{5pt}\begin{tabular}{@{}lrrr@{}}\toprule",
        r"Comparator & Time sign & Treatment sign & Scaled MSE\\\midrule",
    ]
    for design in comparators:
        estimates, intervals = [], []
        for group, metric in ENDPOINTS:
            entry = diff.loc[(design, group, metric)]
            factor = 100 if metric == "direction" else 1
            digits = 2 if metric == "direction" else (5 if site == "tcse" else 3)
            delta, lower, upper = [entry[col] * factor for col in
                                   ["difference", "simultaneous_lower", "simultaneous_upper"]]
            assert lower <= delta <= upper
            estimates.append("$" + f"{delta:.{digits}f}" + "$")
            intervals.append(f"$[{lower:.{digits}f}, {upper:.{digits}f}]$")
        lines.append(" & ".join([NAMES[design], *estimates]) + r"\\")
        lines.append(" & " + " & ".join(intervals) + r"\\[3pt]")
    lines.extend([r"\bottomrule\end{tabular}", r"\par\smallskip\begin{minipage}{.93\linewidth}\footnotesize",
                  "The three column minima among the 18 benchmark designs came from " +
                  ", ".join(NAMES[key] for key in original) + ".",
                  r"\end{minipage}\end{table}"])
    return "\n".join(lines)


def table_recovery(performance):
    values = performance.set_index(["site", "group", "metric", "design"])["mean"].sort_index()
    lines = [
        r"\begin{table}[htbp]\centering\small",
        r"\caption{Temporal sign gain retained and excess magnitude error removed by joint or alternating selection. Percentages use direct sign and matched MSE selection as reference points within each trial. The final column gives the observed treatment sign error.}\label{tab:joint-recovery}",
        r"\setlength{\tabcolsep}{4pt}\begin{tabular}{@{}llrrr@{}}\toprule",
        r"&&Temporal gain&Excess MSE&Treatment sign\\",
        r"Trial & Selection & retained (\%) & removed (\%) & error (\%)\\\midrule",
    ]
    for site, label in SITES.items():
        time = values.loc[(site, "time", "direction")]
        mse = values.loc[(site, "primary", "scaled_mse")]
        treatment = values.loc[(site, "treatment", "direction")]
        temporal_gain = time["contrast_mse"] - time["sign_error"]
        excess_mse = mse["sign_error"] - mse["contrast_mse"]
        for design in ("joint_maximin", "alternate"):
            retained = f"{100 * (time['contrast_mse'] - time[design]) / temporal_gain:.1f}" if temporal_gain else "--"
            removed = f"{100 * (mse['sign_error'] - mse[design]) / excess_mse:.1f}" if excess_mse else "--"
            row = [label, NAMES[design], retained, removed,
                   f"{100 * treatment[design]:.2f}"]
            lines.append(" & ".join(row) + r"\\")
    return "\n".join(lines + [r"\bottomrule\end{tabular}\end{table}"])


def table_soil_example():
    data = pd.read_csv(HERE / "joint_results" / "manure_control.csv")
    comparisons = ["1979 J minus 1979 B", "1989 B minus 1979 B",
                   "1989 J minus 1979 J", "1989 J minus 1989 B"]
    keys = pd.MultiIndex.from_product([["K", "P"], comparisons], names=["element", "comparison"])
    rows = data.set_index(["element", "comparison"])
    assert rows.index.is_unique and rows.index.difference(keys).empty and keys.difference(rows.index).empty
    assert (rows.paid_records == 50).all() and (rows.pilots == 200).all()
    lines = [
        r"\begin{table}[htbp]\centering\small",
        r"\caption{Joint allocation for manure (J) and the cropped unamended control (B) at Ultuna. "
        r"Each allocation uses 50 paid plot-years, including the common pilot. Reference differences "
        r"and root mean squared errors (RMSE) use mg kg$^{-1}$. Sign error is the percentage of "
        r"allocations with a wrong direction across 200 pilots. The alternating rule averages losses "
        r"over both starting orders within each pilot. Table~\ref{tab:ultuna-example} reports the "
        r"single-objective rules for the same comparisons.}\label{tab:joint-soil-example}",
        r"\setlength{\tabcolsep}{4pt}\begin{tabular}{@{}llrrrrr@{}}\toprule",
        r"&&&\multicolumn{2}{c}{Joint maximin gain}&\multicolumn{2}{c}{Alternating criteria}\\",
        r"\cmidrule(lr){4-5}\cmidrule(l){6-7}",
        r"Element&Contrast&Reference&Sign error (\%)&RMSE&Sign error (\%)&RMSE\\\midrule",
    ]
    for (element, comparison), entry in rows.loc[keys].iterrows():
        row = [element, comparison, f"${entry.truth_mg_kg:.1f}$"]
        for design in ["joint_maximin", "alternate"]:
            row.extend([f"{100 * entry[design + '_direction_error']:.2f}",
                        f"{entry[design + '_rmse_mg_kg']:.1f}"])
        lines.append(" & ".join(row) + r"\\")
    return "\n".join(lines + [r"\bottomrule\end{tabular}\end{table}"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("JointTables.tex"))
    args = parser.parse_args()
    performance = primary_frame("performance")
    paired = primary_frame("paired")
    best = primary_frame("best_baselines")
    tables = [table_means(performance)]
    tables.extend(table_differences(paired, best, site) for site in SITES)
    tables.append(table_recovery(performance))
    tables.append(table_soil_example())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n\n".join(tables) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
