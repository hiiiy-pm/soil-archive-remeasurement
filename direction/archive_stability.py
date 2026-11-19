"""Typeset Table S69 from the released archive means and contrast-size sensitivity."""
import argparse
from pathlib import Path

import pandas as pd

from joint_sites_display import ARCHIVES, HERE


def archive_differences():
    sensitivity = pd.read_csv(HERE / "contrast_size_sensitivity.csv")
    temporal = sensitivity[
        (sensitivity.family == "time") & (sensitivity.comparator == "contrast_mse")
    ].set_index("archive")
    if not temporal.index.is_unique:
        raise ValueError("Duplicate temporal sensitivity rows for an archive")

    rows = []
    for name, (site, folder) in ARCHIVES.items():
        scores = pd.read_csv(HERE / folder / "performance.csv")
        period = "budget" if "budget" in scores else "period"
        summary = scores[(scores.site == site) & (scores[period] == "noncensus")]
        means = summary.set_index(["model", "design", "group", "metric"])["mean"]
        if not means.index.is_unique:
            raise ValueError(f"Duplicate archived mean losses for {name}")
        direction = []
        mse_vs_sign = []
        mse_vs_magnitude = []
        for model in ("kriging", "ridge"):
            joint_time = means.loc[model, "joint_maximin", "time", "direction"]
            magnitude_time = means.loc[model, "contrast_mse", "time", "direction"]
            direction.append(100 * (joint_time - magnitude_time))
            joint_mse = means.loc[model, "joint_maximin", "primary", "scaled_mse"]
            direct_mse = means.loc[model, "sign_error", "primary", "scaled_mse"]
            magnitude_mse = means.loc[model, "contrast_mse", "primary", "scaled_mse"]
            mse_vs_sign.append(100 * (joint_mse / direct_mse - 1))
            mse_vs_magnitude.append(100 * (joint_mse / magnitude_mse - 1))

        subset = temporal.loc[name]
        if abs(direction[0] - (subset.all_joint_pct - subset.all_comparator_pct)) > 1e-6:
            raise ValueError(f"All-change sign errors disagree between source tables at {name}")
        above_quartile = subset.above_cutoff_joint_pct - subset.above_cutoff_comparator_pct
        rows.append((name, *direction, above_quartile, *mse_vs_sign, *mse_vs_magnitude))
    return rows


def latex_table(rows):
    lines = [
        r"\begin{table}[!htbp]\centering\small",
        r"\caption{Joint selection across reconstructions and change thresholds. "
        r"Sign error is the joint-minus-magnitude difference in percentage points (pp). "
        r"Combined scaled mean squared error (MSE) is expressed as the percentage change from each comparator. "
        r"Negative values favour joint selection. Entries average nine non-census budgets and 200 shared pilots. "
        r"The post hoc Q1 score uses changes strictly above the within-group first quartile of absolute nonzero differences.}"
        r"\label{tab:archive-stability}",
        r"\renewcommand{\arraystretch}{1.12}\setlength{\tabcolsep}{2.6pt}",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrrr@{}}\toprule",
        r"&\multicolumn{3}{c}{Temporal sign error (pp)}&\multicolumn{4}{c}{Change in MSE (\%)}\\",
        r"\cmidrule(lr){2-4}\cmidrule(l){5-8}",
        r"&\multicolumn{2}{c}{All nonzero}&Above Q1&\multicolumn{2}{c}{vs direct sign}&\multicolumn{2}{c}{vs magnitude}\\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){5-6}\cmidrule(l){7-8}",
        r"Archive & Kriging & Ridge & Kriging & Kriging & Ridge & Kriging & Ridge\\\midrule",
    ]
    for archive, *differences in rows:
        name = archive.replace("Högåsa", r'H\"og\aa sa')
        values = [f"${value:+.2f}$" for value in differences]
        lines.append(" & ".join([name, *values]) + r"\\")
    return "\n".join([*lines, r"\bottomrule\end{tabular*}\end{table}", ""])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(latex_table(archive_differences()), encoding="utf-8")

