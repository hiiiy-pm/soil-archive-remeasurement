"""Render the pilot decision counts and independent-archive accuracy from the numerical release."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from olsen_tables import LABELS, start_table, finish_table, BREAK


ROOT = Path(__file__).resolve().parent / "pilot_olsen_results"
ENDPOINTS = ["treatment", "time", "primary"]
FOCAL = "pilot_olsen"
ORIGINAL = "joint_maximin"
WEIGHTED = "gap_joint"
LABELS = {**LABELS, FOCAL: "Pilot decision"}
INK, GREEN, BLUE, GREY = "#233f4a", "#087975", "#486187", "#cdd5d7"


def main_means(folder, model="kriging"):
    tab = pd.read_csv(folder / "performance.csv")
    tab = tab[(tab["model"] == model) & (tab["period"] == "noncensus")]
    return tab.pivot(index="design", columns="group", values="mean")[ENDPOINTS]


def fields(values):
    return [f"{values.treatment*100:.3f}", f"{values.time*100:.3f}", f"{values.primary:.4f}"]


def primary_table(folder, site_name):
    data = main_means(folder)
    best = data.drop(FOCAL).min()
    txt = start_table("pilot-allocation",
                      f"Choice after the first assays at {site_name}. Signs are percentages of erroneous "
                      "comparisons and magnitude is scaled mean squared error. Values average nine "
                      "non-census budgets over 200 shared pilots under common kriging. The final row "
                      "takes the lowest mean among the 26 alternatives separately for each outcome.",
                      "lrrr",
                      r"Allocation & Treatment signs (\%) & Temporal signs (\%) & Magnitude\\" + r"\midrule")
    for name in [ORIGINAL, WEIGHTED, FOCAL]:
        txt.append(" & ".join([LABELS[name], *fields(data.loc[name])]) + BREAK)
    txt.extend([r"\midrule", " & ".join(["Lowest observed alternative", *fields(best)]) + BREAK])
    return finish_table(txt)


def full_table(folder, site_name):
    common, ridge = [main_means(folder, model) for model in ["kriging", "ridge"]]
    txt = start_table("pilot-all-policies",
                      f"All 27 allocations at {site_name}. For each reconstruction, the first two "
                      "columns give treatment and temporal sign error in percent. The third gives "
                      "scaled mean squared error. The sampling paths and assay costs are shared across reconstructions.",
                      "lrrrrrr",
                      r"&\multicolumn{3}{c}{Kriging}&\multicolumn{3}{c}{Ridge}\\"
                      + "\n" + r"\cmidrule(lr){2-4}\cmidrule(l){5-7}"
                      + "\n" + r"Allocation&Treatment&Time&MSE&Treatment&Time&MSE\\\midrule")
    for name in [FOCAL, ORIGINAL, WEIGHTED, *[n for n in common.index if n not in (FOCAL, ORIGINAL, WEIGHTED)]]:
        txt.append(" & ".join([LABELS[name], *fields(common.loc[name]), *fields(ridge.loc[name])]) + BREAK)
    return finish_table(txt)


def diagnostic_tables(folder, site_name):
    txt = start_table("winchmore-pilot-paired", f"Pilot decision minus fixed or lowest-loss alternatives at {site_name}. "
                      "Directional errors are in percentage points. Intervals simultaneously cover all "
                      "78 primary paired comparisons. Each row uses the same 200 shared pilots.",
                      "lll", r"Outcome & Comparator & Difference [95\% interval]\\\midrule")
    q = pd.read_csv(folder/"paired_differences.csv")
    q = q[(q.model=="kriging") & (q.period=="noncensus")].set_index(["group","design"])
    best = pd.read_csv(folder/"strongest.csv")
    for group,title in zip(ENDPOINTS,["Treatment signs","Temporal signs","Scaled MSE"]):
        benchmarks = best.loc[best.group==group,"design"].tolist()
        alternatives = list(dict.fromkeys([ORIGINAL, WEIGHTED, *benchmarks]))
        for i,name in enumerate(alternatives):
            row = q.loc[(group,name),["mean","lower","upper"]].to_numpy(dtype=float)
            mean,lo,hi = row*(1 if group=="primary" else 100)
            label = LABELS[name] + (" (lowest)" if name in benchmarks else "")
            digits = 3
            while digits < 9 and any(v != 0 and round(v,digits)==0 for v in (mean,lo,hi)):
                digits += 1
            txt.append(" & ".join([title if i==0 else "", label,
                                   f"${mean:+.{digits}f}\\;[{lo:+.{digits}f},{hi:+.{digits}f}]$"]) + BREAK)
        txt.append(r"\addlinespace")
    paired = finish_table(txt)
    txt = start_table("pilot-subgroups",
                      f"Losses of the two fixed courses within each pilot decision at {site_name}. "
                      "The pilot decision uses the loss of the course that it selects. Comparisons "
                      "within each subgroup are descriptive; sign errors are percentages.",
                      "llrrrr", r"Pilot decision & Course evaluated & Pilots & Treatment & Time & MSE\\\midrule")
    frame = pd.read_csv(folder/"subgroup_losses.csv").query("model=='kriging'")
    for chosen in [ORIGINAL, WEIGHTED]:
        subgroup = frame[frame.chosen==chosen]
        n = int(subgroup.n.max())
        for i, design in enumerate([ORIGINAL, WEIGHTED]):
            values = subgroup[subgroup.design==design].set_index("group")["mean"]
            txt.append(" & ".join([LABELS[chosen] if not i else "",LABELS[design],str(n),
                                   *(fields(values) if n else ["--"]*3)]) + BREAK)
    return paired+"\n\n"+finish_table(txt)


def records(folder):
    d = pd.read_csv(folder/"pilot_decisions.csv")
    assert len(d) == 200 and d.pilot.nunique() == 200
    flag = d.chosen == WEIGHTED
    missing = d.informative_domains == 0
    assert not (flag & missing).any()
    return [int(flag.sum()), int((~flag & ~missing).sum()), int(missing.sum())]


def repeated_layer_table(folder):
    tab = pd.read_csv(folder/"domain_results.csv")
    tab = tab.query("model=='kriging' and domain=='P_0-7.5' and family in ['treatment','time']")
    values = tab.pivot(index="design", columns=["metric","family"], values="mean")
    txt = start_table("pilot-topsoil",
                      r"Recovery of Truog P differences in the 0--7.5~cm layer sampled on all "
                      r"17 dates at Winchmore. Each row scores the actual full-archive selection "
                      r"path using only comparisons in the repeated layer. Magnitude is the "
                      r"equally weighted mean of treatment and temporal scaled MSE. The two "
                      r"RMSE columns are in mg P kg$^{-1}$ on the Truog scale.",
                      "lrrrrr",
                      r"&\multicolumn{2}{c}{Wrong sign (\%)}&&\multicolumn{2}{c}{RMSE}\\"
                      + "\n" + r"\cmidrule(lr){2-3}\cmidrule(l){5-6}"
                      + "\n" + r"Allocation & Treatment & Time & Magnitude & Treatment & Time\\\midrule")
    for design in [ORIGINAL,WEIGHTED,FOCAL,"sign_error","contrast_mse","alternate"]:
        row = values.loc[design]
        txt.append(" & ".join([
            LABELS[design],
            *(f"{row['direction',f]*100:.3f}" for f in ["treatment","time"]),
            f"{row['scaled_mse'].mean():.4f}",
            *(f"{np.sqrt(row['native_mse',f]):.3f}" for f in ["treatment","time"]),
        ]) + BREAK)
    return finish_table(txt)


def graph(folder, site_name, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.0,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "text.color": INK,
                         "axes.edgecolor": "#8ca1a7", "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(7.1,5.45),facecolor="white")
    gs = fig.add_gridspec(2,3,left=.17,right=.977,top=.91,bottom=.12,
                          wspace=.64,hspace=.95,height_ratios=[1.08,1])
    ax = fig.add_subplot(gs[0,:])
    colors = [GREEN, BLUE, GREY]
    hatches = [None, "///", None]
    names = ["Park Grass", "Hailun", site_name]
    for i, site in enumerate([ROOT/"parkgrass", ROOT/"hailun", folder]):
        offset = 0
        for j, num in enumerate(records(site)):
            if num:
                ax.barh(2-i,num,left=offset,height=.6,color=colors[j],hatch=hatches[j],
                        edgecolor="white",linewidth=.8)
                if num >= 8:
                    ax.text(offset+num/2,2-i,str(num),ha="center",va="center",
                            color="white" if j < 2 else INK,weight="bold",fontsize=8,
                            bbox=dict(facecolor=colors[j],linestyle="none",pad=.45))
                else:
                    ax.annotate(str(num),(offset+num,2-i),xytext=(3,0),
                                textcoords="offset points",ha="left",va="center",
                                color=INK,weight="bold",fontsize=8)
            offset += num
    ax.set_yticks([2,1,0],names); ax.set_xlim(0,207)
    ax.set_xticks([0,50,100,150,200]); ax.set_xlabel("Number of pilots")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y",length=0)
    ax.set_title("a   Which course did the paid pilot select?",loc="left",fontsize=10,
                 weight="bold", color=INK,pad=11)
    ax.legend(handles=[Patch(facecolor=c,edgecolor="white",hatch=h,label=t) for c,h,t in zip(colors,hatches,
              ["Olsen-weighted joint","Joint selection","Joint selection · uninformative pilot"])],
              loc="upper center",bbox_to_anchor=(.5,-.38),ncol=3,frameon=False,
              fontsize=7.4,columnspacing=1,handlelength=1)
    pair = pd.read_csv(folder/"paired_differences.csv")
    pair = pair[(pair.model=="kriging") & (pair.period=="noncensus")].set_index(["group","design"])
    data = main_means(folder)
    for col,(group,label) in enumerate(zip(ENDPOINTS,["Treatment sign error","Temporal sign error","Magnitude error"])):
        ax=fig.add_subplot(gs[1,col])
        best=data.drop(FOCAL)[group].idxmin()
        factor=100 if group!="primary" else 1
        for y,design,color,mark in zip([2,1,0],[ORIGINAL,WEIGHTED,best],[BLUE,GREEN,"#965724"],["o","s","D"]):
            row=pair.loc[(group,design)]
            mean,lo,hi=row[["mean","lower","upper"]].to_numpy(dtype=float)*factor
            ax.hlines(y,lo,hi,color=color,linewidth=1.65)
            ax.plot(mean,y,marker=mark,markerfacecolor=color,markeredgecolor="white",
                    markeredgewidth=.45,ms=5.2,ls="")
        ax.axvline(0,color="#82939a",ls=(0,(3,2)),lw=.75,zorder=0)
        ax.set_ylim(-.45,2.45); ax.set_yticks([2,1,0],["Joint selection","Fixed weighting","Lowest loss"] if col==0 else ["","",""])
        ax.spines["left"].set_visible(False); ax.tick_params(axis="y",length=0)
        ax.set_title("bcd"[col]+"   "+label,loc="left",fontsize=9,weight="bold",color=INK,pad=8)
        ax.set_xlabel("Difference (pp)" if group!="primary" else "Difference in scaled MSE")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4));ax.margins(x=.13)
    output.mkdir(parents=True,exist_ok=True)
    for ext in ("pdf","eps","png"):
        fig.savefig(output/f"FigS10.{ext}",dpi=300,bbox_inches="tight",pad_inches=.08)
    plt.close(fig)


if __name__ == "__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("site")
    ap.add_argument("--name",required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--tables",type=Path,required=True)
    args=ap.parse_args()
    path=ROOT/args.site
    args.tables.mkdir(exist_ok=True,parents=True)
    (args.tables/"PilotOlsenMain.tex").write_text(primary_table(path,args.name)+"\n",encoding="utf-8")
    tables = full_table(path,args.name)+"\n\n"+diagnostic_tables(path,args.name)+"\n"
    if args.site == "winchmore":
        tables += "\n" + repeated_layer_table(path) + "\n"
    (args.tables/"PilotOlsenTables.tex").write_text(tables,encoding="utf-8")
    graph(path,args.name,args.output)
