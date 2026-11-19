"""Plot paired losses for the joint selector from the complete two-trial summary."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
FOCAL = "joint_maximin"
SITES = {"tcse": "Lincoln-TCSE", "ultuna": "Ultuna"}
COMPARATORS = {
    "alternate": "Fixed alternation",
    "sign_error": "Direct sign error",
    "sign_variance": "Bernoulli variance",
    "contrast_mse": "Contrast MSE",
    "concentration_mse": "Concentration, GP",
    "concentration": "Concentration, ridge",
}
ENDPOINTS = [
    ("time", "direction", "Time direction", "Percentage points", 100, "contrast_mse"),
    ("treatment", "direction", "Treatment direction", "Percentage points", 100, "sign_error"),
    ("primary", "scaled_mse", "Magnitude", "Scaled MSE", 1, "sign_error"),
]
STYLES = {
    "specified": ("#146C70", "^", "#146C70", "Focal comparison"),
    "best": ("#75466D", "D", "#75466D", "Lowest-loss benchmark"),
    "other": ("#43545A", "o", "white", "Other comparison"),
}


def records(performance, paired):
    """Keep one validated simultaneous interval for every displayed difference."""
    summary = performance[(performance.model == "kriging") & (performance.budget == "noncensus")]
    comparisons = paired[(paired.model == "kriging") & (paired.budget == "noncensus") &
                         (paired.focal == FOCAL)]
    panels = {}
    for site in SITES:
        for group, metric, _, _, multiplier, prespecified in ENDPOINTS:
            index = (site, group, metric)
            means = summary[(summary.site == site) & (summary.group == group) &
                            (summary.metric == metric)].set_index("design")["mean"]
            benchmarks = means.drop([FOCAL, "alternate"])
            if not means.index.is_unique or len(benchmarks) != 18:
                raise ValueError(f"Expected 18 distinct benchmark designs for {index}")
            best = benchmarks.idxmin()
            if best not in COMPARATORS:
                raise ValueError(f"Missing lowest-loss benchmark {best!r} for {index}")
            pairs = comparisons[(comparisons.site == site) & (comparisons.group == group) &
                                (comparisons.metric == metric)].set_index("comparator")
            if not pairs.index.is_unique or len(pairs) != 19:
                raise ValueError(f"Incomplete set of paired comparisons for {index}")
            intervals = pairs.loc[list(COMPARATORS), ["difference", "simultaneous_lower",
                                                       "simultaneous_upper"]].copy()
            np.testing.assert_allclose(intervals["difference"], means[FOCAL] - means.loc[intervals.index],
                                       atol=1e-13, rtol=1e-11)
            if not np.isfinite(intervals).all().all() or not (
                (intervals.simultaneous_lower <= intervals.difference) &
                (intervals.difference <= intervals.simultaneous_upper)
            ).all():
                raise ValueError(f"Invalid simultaneous interval for {index}")
            intervals *= multiplier
            intervals["role"] = ["specified" if d == prespecified else "best" if d == best else "other"
                                 for d in intervals.index]
            panels[index] = intervals
    return panels


def limits(frames):
    low = min([0.0] + [frame.simultaneous_lower.min() for frame in frames])
    high = max([0.0] + [frame.simultaneous_upper.max() for frame in frames])
    pad = .08 * (high - low) if high > low else .01
    return low - pad, high + pad


def tick_text(value, _):
    return "0" if abs(value) < 1e-12 else f"{value:.4g}".replace("-", "−")


def draw(performance, paired):
    panels = records(performance, paired)
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.38))
    fig.subplots_adjust(left=.236, right=.984, top=.829, bottom=.189, hspace=.59, wspace=.26)
    rows = np.array([5.65, 4.35, 3.35, 2.35, .95, -.05])
    labels = ["Alternation", "Direct sign", "Sign variance", "Magnitude", "Concentration (GP)", "Concentration (ridge)"]
    headings = ["Temporal sign error", "Treatment sign error", "Difference size"]
    fig.text(.236, .987, "Difference in loss  (joint selection minus comparator)", fontsize=9.4,
             fontweight="bold", color="#20343A", va="top")
    for r, (site, site_name) in enumerate(SITES.items()):
        for c, (group, metric, _, _, _, _) in enumerate(ENDPOINTS):
            ax = axes[r, c]
            table = panels[site, group, metric]
            scale = [panels[s, group, metric] for s in SITES] if metric == "direction" else [table]
            ax.set_xlim(*limits(scale))
            ax.set_ylim(-.53, 6.03)
            ax.axvline(0, color="#48575C", linewidth=.9, linestyle=(0, (3, 2)), zorder=1)
            for row_pos, (_, item) in zip(rows, table.iterrows()):
                color, marker, face, _ = STYLES[item.role]
                ax.errorbar(item.difference, row_pos,
                            xerr=[[item.difference - item.simultaneous_lower],
                                  [item.simultaneous_upper - item.difference]],
                            fmt=marker, markersize=4.8, capsize=2.0, elinewidth=1.15,
                            color=color, markerfacecolor=face, markeredgewidth=.9, zorder=3)
            ax.set_yticks(rows, labels if c == 0 else [])
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#E3E8E8", linewidth=.5)
            ax.tick_params(axis="y", length=0, pad=5.5, labelsize=9)
            ax.tick_params(axis="x", length=2.7, width=.6, color="#737F83", pad=3, labelsize=8.8)
            nbins = 4 if metric == "direction" else 3
            ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 5, 10], min_n_ticks=2))
            ax.xaxis.set_major_formatter(FuncFormatter(tick_text))
            for spine in ("top", "left", "right"):
                ax.spines[spine].set_visible(False)
            ax.spines["bottom"].set(color="#869196", linewidth=.65)
            letter = "abcdef"[r * 3 + c]
            ax.set_title(letter, loc="left", fontsize=10.0, fontweight="bold", color="#20343A", pad=4)
            if c == 0:
                ax.text(-.06, 1.045, site_name, transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=10.0, fontweight="bold", color="#20343A")
            if r == 0:
                ax.text(0, 1.255, headings[c], transform=ax.transAxes, fontsize=9.8, weight="bold",
                        color="#20343A", va="center")
            if r == 1:
                ax.set_xlabel("Percentage points" if c < 2 else "Scaled MSE", fontsize=9.1, labelpad=3)
    handles = [Line2D([], [], color=color, marker=marker, markerfacecolor=face, linewidth=1.1,
                      markersize=4.5, markeredgewidth=.9, label=label)
               for color, marker, face, label in STYLES.values()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .007), ncol=3,
               fontsize=9.0, frameon=False, columnspacing=1.2, handlelength=1.5, handletextpad=.4)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE.parent / "artwork")
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.6,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    results = HERE / "joint_results"
    perf = pd.read_csv(results / "performance.csv", dtype={"budget": str})
    paired = pd.read_csv(results / "paired.csv", dtype={"budget": str})
    figure = draw(perf, paired)
    args.output.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "eps", "png"):
        figure.savefig(args.output / f"Fig3.{ext}", dpi=400, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
