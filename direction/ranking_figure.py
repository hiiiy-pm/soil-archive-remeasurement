"""Figure 4. Where the Lanna ranking loss arose and how the fixed rule compares."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
FIELDS = ["P_High", "P_Low"]
PAIRS = ["Control minus P", "Control minus P6", "P minus P6"]
SITE_NAMES = {"tcse": "Lincoln-TCSE", "ultuna": "Ultuna", "lanna": "Lanna"}


def diagnostic_panels(fig):
    data = pd.read_csv(HERE/"ranking_diagnosis/contrast_losses.csv").query("family == 'treatment'")
    assert len(data) == 34 and data.true_sign.ne(0).all()
    palette = LinearSegmentedColormap.from_list("change", ["#146C70", "#F6F6F2", "#93451C"])
    extent = np.ceil(data.joint_minus_direct.abs().max()*100)
    scale = Normalize(-extent, extent)
    top = fig.add_gridspec(1,2, left=.224, right=.986, top=.821, bottom=.652,
                           width_ratios=[7,5], wspace=.085)
    for j,field in enumerate(FIELDS):
        ax=fig.add_subplot(top[j])
        grid=(data[data.domain==field].pivot(index="treatment_pair",columns="first_year",
                                             values="joint_minus_direct").reindex(PAIRS)*100)
        ax.imshow(np.ma.masked_invalid(grid), cmap=palette,norm=scale,aspect="auto")
        ax.set_facecolor("#D6DCDF")
        for row,col in np.ndindex(grid.shape):
            val=grid.iat[row,col]
            label="—" if np.isnan(val) else ("0.0" if abs(val)<.05 else f"{val:+.1f}".replace("-", "\N{MINUS SIGN}"))
            rgb = np.array(palette(scale(val))[:3]) if np.isfinite(val) else np.array([.84, .86, .87])
            linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
            luminance = linear @ np.array([.2126, .7152, .0722])
            black_contrast = (luminance + .05) / .05
            white_contrast = 1.05 / (luminance + .05)
            ax.text(col, row, label, ha="center", va="center", fontsize=9.0,
                    color="black" if black_contrast >= white_contrast else "white")
        ax.set(xticks=range(grid.shape[1]),xticklabels=grid.columns,yticks=range(3))
        ax.set_yticklabels(["No P – annual P","No P – six-year P","Annual – six-year P"] if j==0 else [])
        ax.tick_params(length=0,pad=3,labelsize=8.8)
        ax.set_xticks(np.arange(-.5,grid.shape[1],1),minor=True)
        ax.set_yticks(np.arange(-.5,3,1),minor=True)
        ax.grid(which="minor",color="white",linewidth=1.2)
        ax.tick_params(which="minor",length=0)
        ax.set_title(["High field half","Low field half"][j],fontsize=9.3,weight="bold",color="#20343A",loc="left")
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(.025,.958,"a   Treatment rankings at Lanna", fontsize=11, weight="bold", color="#20343A")
    fig.text(.025,.907,"Change in sign error, pooled joint minus direct sign selection (percentage points)",
             fontsize=9.1, color="#34484E")


def endpoint_panels(fig):
    perf=pd.read_csv(HERE/"ranking_results/performance.csv",dtype={"budget":str})
    diffs=pd.read_csv(HERE/"ranking_results/paired.csv",dtype={"budget":str})
    select=lambda d: d[(d.model=="kriging") & (d.budget=="noncensus")]
    means=select(perf).set_index(["site","group","metric","design"])["mean"]
    paired=select(diffs).query("focal == 'split_direct'").set_index(["site","group","metric","comparator"])
    specs=[("time","direction","contrast_mse","b   Temporal signs"),
           ("treatment","direction","sign_error","c   Treatment signs"),
           ("primary","scaled_mse","sign_error","d   Magnitude")]
    layout=fig.add_gridspec(1,3,left=.175,right=.98,bottom=.152,top=.452,wspace=.31)
    for j,(group,metric,ref,title) in enumerate(specs):
        ax=fig.add_subplot(layout[0,j])
        for i,site in enumerate(SITE_NAMES):
            scale=100. if j<2 else 100./means.loc[(site,group,metric,"joint_maximin")]
            for other,dy,color,mark in [("joint_maximin",.14,"#146C70","o"),(ref,-.14,"#93451C","D")]:
                row=paired.loc[(site,group,metric,other)]
                val,lo,hi=[scale*row[k] for k in ("difference","simultaneous_lower","simultaneous_upper")]
                y=2-i+dy
                ax.errorbar(val, y, xerr=[[val-lo], [hi-val]], fmt=mark, ms=4.6, color=color,
                            mfc=color if other=="joint_maximin" else "white", mec=color, mew=.95,
                            capsize=2.0, elinewidth=1.2, zorder=3)
        ax.axvline(0, color="#48575C", lw=.9, ls=(0, (3, 2)))
        for y in [.5,1.5]:
            ax.axhline(y,color="#DFE6E7",lw=.6)
        ax.set(yticks=[2,1,0],ylim=(-.48,2.5),xlabel="Percentage points" if j<2 else "% of pooled-joint MSE")
        ax.set_yticklabels(list(SITE_NAMES.values()) if j==0 else [])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4,min_n_ticks=3))
        ax.tick_params(axis="y",length=0)
        ax.tick_params(axis="x",length=2.4,pad=2)
        ax.set_title(title,fontsize=9.7,weight="bold",loc="left",color="#20343A",pad=6)
        for side in ["top","right","left"]:
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#839098")
    fig.text(.025, .536, "Difference in loss, separate sign risks minus comparator",
             fontsize=9.4, weight="bold", color="#20343A")
    handles = [Line2D([], [], marker="o", color="#146C70", lw=1.2, ms=4.6),
               Line2D([], [], marker="D", color="#93451C", mfc="white", lw=1.2, ms=4.6)]
    fig.legend(handles, ["Versus pooled joint", "Versus single-objective reference"], loc="lower center",
               bbox_to_anchor=(.52, .018), ncol=2, frameon=False, fontsize=9.1,
               handlelength=1.5, columnspacing=1.9)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("artwork"))
    args=parser.parse_args()
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8.8,"pdf.fonttype":42,"ps.fonttype":42})
    fig=plt.figure(figsize=(7.0,5.03))
    diagnostic_panels(fig)
    endpoint_panels(fig)
    args.output.mkdir(parents=True,exist_ok=True)
    for ext in ["png","eps","pdf"]:
        fig.savefig(args.output/f"Fig4.{ext}",dpi=600,facecolor="white")
    plt.close(fig)
