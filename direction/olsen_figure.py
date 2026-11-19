"""Plot observed archived chemistry and paired changes in recovery under gap weighting."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np
import pandas as pd

from trials import ROOT


RESULTS = ROOT / "direction" / "olsen_results"
SITES = [("parkgrass", "Park Grass"), ("hailun", "Hailun")]
COLORS = ["#086963", "#3e527f"]
INK = "#223b48"
GRID = "#dbe3e4"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "text.color": INK,
                     "axes.labelcolor": INK, "axes.edgecolor": "#95a6ab",
                     "xtick.color": INK, "ytick.color": INK})


def panel_heading(axis, letter, title):
    axis.set_title(f"{letter}   {title}", loc="left", fontsize=9.6,
                   color=INK, weight="bold", pad=10)
    axis.tick_params(length=2.5, labelsize=7.5, width=.65)
    axis.set_axisbelow(True)


def chemistry(axis, site, title, idx):
    frame = pd.read_csv(RESULTS / site / "frozen_gaps.csv")
    panel_heading(axis, "ab"[idx], title)
    axis.axhline(1, color="#879a9e", lw=.8, ls=(0,(3,3)))
    for domain, g in frame.groupby("domain", sort=False):
        g = g.sort_values("gap")
        shallow = (site == "hailun" and len(g) == 3)
        color = "#a65c21" if shallow else COLORS[idx]
        marker = "^" if shallow else ("o" if idx == 0 else "s")
        label = ("0–10 cm" if shallow else "0–20 cm") if idx else "0–23 cm"
        axis.plot(g.gap, g.multiplier, color=color, lw=.95, zorder=2)
        axis.scatter(g.gap, g.multiplier, marker=marker, c=color,
                     s=(24 if shallow else 10 if idx else 4), linewidths=0,
                     label=f"{label}  ·  {len(g)}", zorder=3)
    axis.set_xscale("symlog", linthresh=1, linscale=.75)
    ticks = [x for x in [0, 1, 3, 10, 30, 100, 300, 1000] if x <= 1.1*frame.gap.max()]
    axis.set_xlim(-.08, max(1.1*frame.gap.max(), 2))
    axis.set_xticks(ticks, [str(x) for x in ticks])
    axis.set_ylim(-.05, 2.11)
    axis.set_yticks([0, .5, 1, 1.5, 2])
    axis.set_xlabel("Absolute Olsen P gap (mg P kg⁻¹)")
    if not idx:
        axis.set_ylabel("Treatment sign weight multiplier")
    axis.legend(loc="upper right", frameon=False, fontsize=7.1, title="Depth · comparisons",
                title_fontsize=7.2, borderaxespad=.15, handletextpad=.1)


def outcome(axis, group, title, letter):
    estimates = []
    folders = [RESULTS/site for site, _ in SITES]
    folders.append(ROOT/"direction/pilot_olsen_results/winchmore")
    for folder in folders:
        paired = pd.read_csv(folder / "paired_differences.csv")
        primary = paired.query("model == 'kriging' and period == 'noncensus' and design == 'joint_maximin'")
        interval = primary.loc[primary.group == group, ["mean","lower","upper"]].iloc[0]
        if group == "primary":
            performance = pd.read_csv(folder / "performance.csv")
            original = performance.query("model == 'kriging' and period == 'noncensus' and "
                                         "design == 'joint_maximin' and group == 'primary'")["mean"].item()
            interval /= original
        estimates.append(interval.to_numpy()*100)
    panel_heading(axis, letter, title)
    axis.axvline(0, color="#8c9ea2", lw=.9, ls=(0,(3,3)))
    lower = min(0, *(r[1] for r in estimates))
    upper = max(0, *(r[2] for r in estimates))
    span = max(upper-lower, 1e-8)
    axis.set_xlim(lower-.23*span, upper+.23*span)
    for pos, (estimate, lo, hi), color, marker in zip([2,1,0], estimates, [*COLORS, "#9c511e"], ["o","s","D"]):
        axis.plot([lo,hi], [pos,pos], lw=1.7, color=color)
        axis.scatter(estimate,pos,s=32,c=color,marker=marker, edgecolors="white",linewidths=.5,zorder=4)
        digits = 3 if 0 < abs(estimate) < .02 else 2
        axis.text(estimate, pos+.20, f"{estimate:+.{digits}f}".replace("-", "−"),
                  color=color, ha="center", fontsize=7.5)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
    fmt = ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)
    axis.xaxis.set_major_formatter(fmt)
    axis.set_yticks([2,1,0], ["Park Grass · fixed","Hailun · fixed","Winchmore · pilot"] if letter == "c" else ["","",""])
    axis.set_ylim(-.45,2.55)
    axis.spines["left"].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.set_xlabel("Change (% of joint MSE)" if group == "primary" else "Change (percentage points)")


def make(output):
    fig = plt.figure(figsize=(7.05, 5.42), facecolor="white")
    axes = fig.add_gridspec(2,6,left=.115,right=.98,top=.90,bottom=.105,
                            hspace=.83,wspace=1.55,height_ratios=[1.13,1.0])
    for idx, (site, title) in enumerate(SITES):
        chemistry(fig.add_subplot(axes[0, idx*3:(idx+1)*3]),site,title,idx)
    for idx, (group, title) in enumerate([("treatment","Treatment signs"),("time","Temporal signs"),
                                          ("primary","Magnitude error")]):
        outcome(fig.add_subplot(axes[1,idx*2:(idx+1)*2]),group,title,"cde"[idx])
    fig.text(.115,.983,"BEFORE REMEASUREMENT",color="#596e77",fontsize=7.6,weight="bold",va="top")
    fig.text(.115,.485,"CHANGE FROM JOINT SELECTION WITHOUT OLSEN WEIGHTING",color="#596e77",
             fontsize=7.6,weight="bold",va="top")
    output.mkdir(parents=True,exist_ok=True)
    for ext in ("pdf", "eps", "png"):
        fig.savefig(output/f"FigS11.{ext}", facecolor="white", dpi=330, bbox_inches="tight",pad_inches=.06)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"artwork")
    make(parser.parse_args().output)
