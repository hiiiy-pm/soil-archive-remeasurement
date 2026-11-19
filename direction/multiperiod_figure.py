"""Draw the Lanna archive and equal-budget allocation curves for Figure S5."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from multiperiod_trial import SITE, load_trial
from multiperiod_summary import OUT


OBJECTIVES = [("time", "direction", "Temporal signs", "Wrong signs (%)"),
              ("treatment", "direction", "Treatment signs", "Wrong signs (%)"),
              ("primary", "scaled_mse", "Magnitude", "Scaled MSE")]
STYLES = {"joint_maximin": ("#00786F", "-", "o", "Joint"),
          "alternate": ("#7E3F70", "-", "s", "Alternating"),
          "sign_error": ("#9E511D", "--", "^", "Direct sign"),
          "contrast_mse": ("#345C80", "-.", "D", "Matched MSE")}
TREATMENTS = [("Control", "#435A68", "o"), ("P", "#AF5B20", "s"), ("P6", "#126E5F", "^")]


def clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2.4, pad=2)
    ax.grid(axis="y", color="#DCE2E5", linewidth=.55, zorder=0)
    ax.set_axisbelow(True)


def plot(output, folder):
    frame, panels, _, responses, _ = load_trial(SITE)
    frame = frame.copy()
    for panel in panels:
        frame.loc[panel.eligible, "reference"] = responses[panel.name][panel.eligible]
    assert frame.reference.notna().all()
    perf = pd.read_csv(folder / "performance.csv", dtype={"budget": str})
    perf = perf[perf.model.eq("kriging")]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.7,
                         "axes.edgecolor": "#94A1A8", "axes.labelcolor": "#26343A",
                         "xtick.color": "#415057", "ytick.color": "#415057",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(7.12, 5.3))
    topo = fig.add_gridspec(1, 2, left=.085, right=.984, bottom=.626, top=.88, wspace=.19)
    lower = fig.add_gridspec(1, 3, left=.085, right=.984, bottom=.115, top=.385, wspace=.45)
    fields = sorted(frame.field.unique())
    trials = []
    for j, field in enumerate(fields):
        ax = fig.add_subplot(topo[0, j], sharey=trials[0] if trials else None)
        trials.append(ax)
        d = frame[frame.field == field]
        for tr, color, marker in TREATMENTS:
            samples = d[d.treatment == tr]
            assert not samples.empty
            means = samples.groupby("year").reference.mean()
            ax.plot(means.index, means, color=color, marker=marker, linewidth=1.05, ms=3.8)
            last = samples[samples.year.eq(2021)].reference
            ax.plot([2021, 2021], [last.min(), last.max()], color=color, linewidth=1.4)
        clean(ax)
        ax.set(xlim=(1965, 2024), xticks=[1970, 1990, 2010, 2021])
        ax.set_title(f"{chr(97+j)}   {field} field half", loc="left", fontweight="bold", fontsize=8.7)
        if j == 0:
            ax.set_ylabel("Reference P (mg/kg)")
        else:
            ax.tick_params(labelleft=False)
    fig.text(.085, .966, "Measured phosphorus across the archive", fontsize=10.2, weight="bold")
    fig.legend([Line2D([0], [0], color=c, marker=m, lw=1.1, ms=4) for _,c,m in TREATMENTS],
               ["No P addition", "P yearly", "P every sixth year"], loc="upper right",
               bbox_to_anchor=(.994, .953), frameon=False, ncol=3, columnspacing=1.2)
    fig.text(.085,.503,"Recovery as the assay budget increases", fontsize=10.2, weight="bold")
    series = [(k, *v) for k,v in STYLES.items()]
    handles = [Line2D([], [], color=c, ls=ls, marker=m, ms=3.5, lw=1.3)
               for _,c,ls,m,_ in series]
    fig.legend(handles, [v[-1] for v in series], loc="upper left", bbox_to_anchor=(.075,.497),
               ncol=4, frameon=False, columnspacing=1.2)
    for j,(group,metric,title,unit) in enumerate(OBJECTIVES):
        ax = fig.add_subplot(lower[0,j])
        group_rows = perf[(perf.group==group) & (perf.metric==metric)]
        for method,color,style,marker,_ in series:
            rows = group_rows[(group_rows.design==method) & group_rows.budget.str.isdigit()].copy()
            rows["assays"] = rows.budget.astype(int)
            rows.sort_values("assays", inplace=True)
            assert len(rows)==10
            ax.plot(rows.assays,rows["mean"]*(1 if j==2 else 100),color=color,ls=style,marker=marker,
                    ms=3, lw=1.25, markerfacecolor="white" if method=="alternate" else color)
        clean(ax)
        ax.set(xlabel="Paid assays", ylabel=unit, xticks=[4,12,21,29,41], ylim=(0,None))
        ax.set_title(f"{chr(99+j)}   {title}",loc="left",fontweight="bold",fontsize=8.7)
    output.mkdir(parents=True,exist_ok=True)
    for ext in ("pdf", "eps", "png"):
        fig.savefig(output/f"FigS5.{ext}",dpi=390,facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artwork"))
    parser.add_argument("--results", type=Path, default=OUT / SITE)
    args = parser.parse_args()
    plot(args.output,args.results)
