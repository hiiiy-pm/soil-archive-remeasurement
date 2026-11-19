"""Draw the cross-archive figure and typeset its absolute losses from stored results."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ARCHIVES = {
    "Lincoln-TCSE": ("tcse", "joint_results"),
    "Ultuna": ("ultuna", "joint_results"),
    "Lanna": ("lanna", "multiperiod_results/lanna"),
    "Högåsa": ("hogasa", "external_results/hogasa"),
    "Park Grass": ("parkgrass", "olsen_results/parkgrass"),
    "Hailun": ("hailun", "olsen_results/hailun"),
    "Winchmore": ("winchmore", "pilot_olsen_results/winchmore"),
}


def site_means():
    rows = []
    for name, (site, folder) in ARCHIVES.items():
        data = pd.read_csv(HERE / folder / "performance.csv")
        period = "budget" if "budget" in data else "period"
        main = data[(data.site == site) & (data.model == "kriging") & (data[period] == "noncensus")]
        means = main.set_index(["design","group","metric"])["mean"]
        row = {"Archive": name}
        for outcome, group, metric in [("Time", "time", "direction"),
                                       ("MSE", "primary", "scaled_mse"),
                                       ("Treatment", "treatment", "direction")]:
            scale = 100 if metric == "direction" else 1
            for label, rule in [("magnitude", "contrast_mse"),
                                ("joint", "joint_maximin"),
                                ("sign", "sign_error")]:
                row[f"{outcome}_{label}"] = means.loc[rule, group, metric] * scale
        rows.append(row)
    return pd.DataFrame(rows).set_index("Archive")


def table(data):
    lines = [
        r"\begin{table}[!htbp]\centering\small",
        r"\caption{Recovery of laboratory differences at equal budgets under magnitude selection (Magn.), joint selection (Joint) and direct sign selection (Sign). Lower values indicate better recovery. Scaled mean squared error (MSE) gives equal weight to supported treatment and temporal comparison families within a domain. Each cell averages the nine partial budgets and 200 shared pilots under common kriging. Text comparisons use unrounded values.}\label{tab:joint-sites}",
        r"\renewcommand{\arraystretch}{1.15}\setlength{\tabcolsep}{3pt}\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l*{9}{r}@{}}\toprule",
        r"&\multicolumn{3}{c}{Temporal sign error (\%)}&\multicolumn{3}{c}{Scaled MSE}&\multicolumn{3}{c}{Treatment sign error (\%)}\\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(l){8-10}",
        r"Archive & Magn. & Joint & Sign & Magn. & Joint & Sign & Magn. & Joint & Sign\\\midrule",
    ]
    for name, row in data.iterrows():
        vals = []
        for column, value in row.items():
            digits = 2
            if column.startswith("MSE"):
                digits = 5 if value < 1 else 3
            vals.append(f"{value:.{digits}f}")
        tex = name.replace("Högåsa",r'H\"og\aa sa')
        lines.append(" & ".join([tex,*vals]) + r"\\")
    return "\n".join([*lines,r"\bottomrule\end{tabular*}\end{table}",""])


def figure(data, output):
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.0,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.spines.left": False,
                         "axes.edgecolor": "#7C898E", "text.color": "#20343A",
                         "xtick.color": "#35474C", "ytick.color": "#35474C"})
    fig = plt.figure(figsize=(7.0, 4.03))
    outcomes = [
        ("a   Temporal direction", "vs magnitude selection", "Percentage points",
         data.Time_joint - data.Time_magnitude),
        ("b   Difference size", "vs direct sign selection", "Change in MSE (%)",
         100 * (data.MSE_joint / data.MSE_sign - 1)),
        ("c   Treatment ranking", "vs direct sign selection", "Percentage points",
         data.Treatment_joint - data.Treatment_sign),
    ]
    positions = np.arange(len(data))[::-1]
    lower, higher = "#146C70", "#93451C"
    for idx, (title, baseline, unit, differences) in enumerate(outcomes):
        left = .157 + .285 * idx
        ax = fig.add_axes([left, .222, .179, .548])
        fig.text(left, .906, title, va="center", weight="bold", fontsize=10.1)
        fig.text(left, .844, baseline, va="center", fontsize=8.8, color="#38494E")
        for y, change in zip(positions, differences):
            color = lower if change < 0 else higher
            ax.axhline(y, color="#E1E7E8", lw=.55, zorder=0)
            ax.hlines(y, 0, change, color=color, lw=1.8, zorder=2)
            ax.plot(change, y, marker="o" if change < 0 else "D", ms=4.7, mew=.8,
                    mfc=color if change < 0 else "white", mec=color, zorder=3)
            signed_value = f"{change:+.2f}".replace("-", "−")
            ax.text(1.39, y, signed_value, transform=ax.get_yaxis_transform(),
                    fontsize=9.0, ha="right", va="center", color=color)
        ax.axvline(0, color="#48575C", lw=.9, ls=(0, (3, 2)), zorder=1)
        low, high = min(0, differences.min()), max(0, differences.max())
        padding = .09 * (high - low)
        ax.set_xlim(low - padding, high + padding)
        ax.set_ylim(-.52, len(positions) - .48)
        ax.set_xlabel(unit, fontsize=8.9, labelpad=3.5)
        ax.set_yticks(positions, data.index if idx == 0 else [])
        ax.tick_params(axis="y", length=0, pad=5, labelsize=9.1)
        ax.tick_params(axis="x", length=2.6, width=.6, labelsize=8.8)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]))
    handles = [Line2D([], [], marker="o", color=lower, lw=1.8, ms=4.7),
               Line2D([], [], marker="D", color=higher, mfc="white", lw=1.8, ms=4.7)]
    fig.legend(handles, ["Lower loss with joint selection", "Higher loss with joint selection"],
               loc="lower center", bbox_to_anchor=(.52, .018), ncol=2, frameon=False,
               fontsize=9.0, columnspacing=2, handlelength=1.7, handletextpad=.5)
    output.mkdir(exist_ok=True, parents=True)
    for suffix in ["pdf", "eps", "png"]:
        fig.savefig(output / f"Fig5.{suffix}", facecolor="white", dpi=330)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--table",type=Path,required=True)
    args = parser.parse_args()
    means = site_means()
    args.table.parent.mkdir(exist_ok=True,parents=True)
    args.table.write_text(table(means),encoding="utf-8")
    figure(means,args.output)
