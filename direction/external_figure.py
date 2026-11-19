"""Draw selection frequencies, realized decisions and paired losses at Högåsa."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, FuncFormatter
import numpy as np
import pandas as pd

from joint_tables import NAMES


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "external_results" / "hogasa"
ENDPOINTS = [
    ("time", "direction", "Temporal sign", "Percentage points", 100),
    ("treatment", "direction", "Treatment sign", "Percentage points", 100),
    ("primary", "scaled_mse", "Magnitude", "Scaled MSE", 1),
]
FIXED = ["joint_maximin", "alternate"]
POLICY_COLORS = ["#047978", "#82476e"]
OUTCOME_COLORS = ["#345c72", "#e4eaec", "#995623"]
INK = "#192e35"
BASELINE_NAMES = {
    "sign_error": "Direct sign error", "sign_variance": "Bernoulli variance",
    "contrast_mse": "Matched contrast MSE", "all_contrast_mse": "All-family contrast MSE",
    "conversion_mse": "Kriging conversion", "concentration_mse": "Kriging concentration",
    "a4_loo_targetcell": "Four-wave target-cell",
}


def primary(results, name):
    df = pd.read_csv(results / (name + ".csv"), dtype={"budget": str})
    return df[(df.site == "hogasa") & (df.budget == "noncensus") & (df.model == "kriging")]


def draw(results):
    choices = pd.read_csv(results / "pilot_choices.csv")
    quality = primary(results, "decision_quality").set_index(["group", "metric"])
    best = primary(results, "best_baselines").set_index(["group", "metric"])
    paired = primary(results, "paired")
    performance = primary(results, "performance").set_index(["group", "metric", "design"])["mean"]
    total = len(choices)
    if (total != 200 or choices.repeat.nunique() != total or not choices.choice.isin(FIXED).all()
            or paired.simultaneous_lower.notna().sum() != 60):
        raise ValueError("Expected all 200 pilots and 60 simultaneous paired intervals.")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelcolor": INK,
                         "text.color": INK, "xtick.color": INK, "ytick.color": INK,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(7.05, 4.8))
    ax_choice = fig.add_axes([.15, .68, .26, .19])
    ax_quality = fig.add_axes([.594, .665, .376, .225])
    fig.text(.022, .944, "a   Choice from the pilot", fontsize=10, weight="bold")
    fig.text(.458, .944, "b   Realized loss versus unchosen rule", fontsize=10, weight="bold")
    for i, (key, color) in enumerate(zip(FIXED, POLICY_COLORS)):
        count = int((choices.choice == key).sum())
        y = 1 - i
        ax_choice.barh(y, count, height=.42, color=color, zorder=2)
        ax_choice.text(count + 4, y, str(count), va="center", fontsize=8.4)
    ax_choice.set_yticks([1, 0], ["Joint", "Alternating"])
    ax_choice.set_ylim(-.56, 1.56)
    ax_choice.set_xlim(0, total * 1.13)
    ax_choice.set_xticks([0, total // 2, total])
    ax_choice.set_xlabel("Pilots (n = 200)", labelpad=2)

    for j, (group, metric, _, _, _) in enumerate(ENDPOINTS):
        q = quality.loc[(group, metric)]
        counts = q[["chosen_strictly_better", "chosen_tied", "chosen_worse"]].to_numpy(dtype=int)
        assert counts.sum() == total
        left = 0
        for n, color in zip(counts, OUTCOME_COLORS):
            width = n * 100 / total
            ax_quality.barh(2 - j, width, left=left, color=color, height=.53,
                            edgecolor="white", linewidth=.4)
            if width >= 14:
                ax_quality.text(left + width / 2, 2 - j, f"{width:g}", va="center", ha="center",
                                fontsize=7.5, color=INK if color == OUTCOME_COLORS[1] else "white")
            left += width
    ax_quality.set_ylim(-.55, 2.55)
    ax_quality.set_xlim(0, 100)
    ax_quality.set_yticks([2, 1, 0], [e[2] for e in ENDPOINTS])
    ax_quality.set_xticks([0, 50, 100])
    ax_quality.set_xlabel("Share of pilots (%)", labelpad=2)
    ax_quality.legend(handles=[Patch(facecolor=c, label=l) for l, c in
                              zip(("Lower", "Tied", "Higher"), OUTCOME_COLORS)],
                      loc="upper center", bbox_to_anchor=(.5, -0.35), ncol=3, frameon=False,
                      fontsize=7.5, handlelength=.85, handletextpad=.35, columnspacing=1.0)
    for ax in (ax_choice, ax_quality):
        ax.tick_params(axis="y", length=0, pad=3)
        ax.tick_params(axis="x", length=2.5, width=.6, pad=2)
        for side in ("left", "right", "top"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#98a6ab")

    locations = [.148, .438, .728]
    for j, (group, metric, title, unit, scale) in enumerate(ENDPOINTS):
        ax = fig.add_axes([locations[j], .176, .25, .246])
        ax.set_title("cde"[j] + "   " + title, loc="left", fontweight="bold", fontsize=9.6, pad=18)
        gate = performance.loc[(group, metric, "gate")] * scale
        gate_label = f"{gate:.2f}%" if metric == "direction" else f"{gate:.3g}"
        ax.text(0, 1.045, "Pilot rule " + gate_label, transform=ax.transAxes, fontsize=7.8)
        best_id = best.loc[(group, metric), "best_original"]
        rows = paired[(paired.group == group) & (paired.metric == metric)].set_index("comparator")
        low, high = 0., 0.
        for k, (name, color, symbol) in enumerate(zip(FIXED + [best_id],
                                                      POLICY_COLORS + ["#495b65"], ["o", "s", "D"])):
            row = rows.loc[name]
            x, left, right = scale * row[["difference", "simultaneous_lower",
                                           "simultaneous_upper"]].to_numpy(dtype=float)
            np.testing.assert_allclose(x, gate - performance.loc[(group, metric, name)] * scale,
                                       atol=1e-11)
            assert np.isfinite([left, right]).all() and left <= x <= right
            low, high = min(low, left), max(high, right)
            ax.hlines(2 - k, left, right, color=color, linewidth=1.4)
            ax.plot(x, 2 - k, marker=symbol, color=color, markersize=4.6)
        pad = .09 * (high - low) if high > low else .01
        ax.set_xlim(low - pad, high + pad)
        ax.set_ylim(-.55, 2.55)
        ax.set_yticks([2, 1, 0], ["Fixed joint", "Fixed alternation", "Best baseline"] if j == 0 else [])
        ax.axvline(0, color=INK, linewidth=.75, zorder=0)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e1e7e8", linewidth=.6)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10], min_n_ticks=2))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, _: "0" if val == 0
                                                    else f"{val:.3g}".replace("-", "−")))
        ax.tick_params(axis="y", length=0, pad=3)
        ax.tick_params(axis="x", length=2.5, width=.6, pad=2)
        for side in ("top", "left", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#98a6ab")
        ax.set_xlabel(unit, fontsize=7.7, labelpad=2)
        best_name = BASELINE_NAMES.get(best_id, NAMES[best_id]).replace("--", "–")
        ax.text(.5, -.38, best_name, transform=ax.transAxes, ha="center", va="top", fontsize=7.1)
    fig.text(.013, .082, "Best baseline", fontsize=7.1, va="center")
    fig.text(.56, .020, "Difference in loss (pilot rule minus comparator)", ha="center", fontsize=8.3)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artwork"))
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    figure = draw(args.results)
    for suffix in ("pdf", "eps", "png"):
        figure.savefig(args.output / ("FigS6." + suffix), dpi=400, facecolor="white")
    plt.close(figure)
