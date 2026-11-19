"""Plot the independent pilot-warning test and typeset its supplementary results."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from joint_tables import NAMES


HERE = Path(__file__).resolve().parent
FOCAL = "pilot_warning"
ORIGINAL = "joint_maximin"
LABELS = dict(NAMES, joint_maximin="Joint selection", pilot_warning="Pilot warning",
              split_direct="Separate sign risks", pooled_direct="Pooled sign risk",
              split_bernoulli="Separate sign variances")
OUTCOMES = [("treatment", "Treatment signs", 100), ("time", "Temporal signs", 100),
            ("primary", "Magnitude", 1)]
TEAL, RUST, INK, LIGHT = "#096E6B", "#9B5133", "#21343B", "#D9E0E1"
ROW = r"\\"


def read_results(results):
    def primary_file(name):
        frame = pd.read_csv(results / name, dtype={"period": str})
        return frame[(frame.model == "kriging") & (frame.period == "noncensus")]
    means = primary_file("performance.csv").set_index(["group", "design"])["mean"]
    pairs = primary_file("warning_minus_comparator.csv").set_index(["group", "design"])
    diagnostic = pd.read_csv(results / "warning_discrimination.csv").set_index(["period", "metric"])
    budgets = pd.read_csv(results / "warning_by_pilot_budget.csv")
    sensitivity = pd.read_csv(results / "warning_sensitivity.csv", dtype={"period": str})
    sensitivity = sensitivity.set_index(["period", "eligible_comparisons"])
    assert len(means) == 72 and len(pairs) == 69 and budgets.pilot.nunique() == 200
    assert (pairs.n == 200).all()
    return means, pairs, diagnostic, budgets, sensitivity


def best_design(means, group, eligible):
    return means.loc[group].reindex(sorted(eligible)).idxmin()


def render_figure(data, out):
    means, pairs, diagnostic, budgets, sensitivity = data
    mean_by_budget = budgets.groupby("budget")[["high_error", "low_error"]].mean() * 100
    x = mean_by_budget.index.to_numpy()
    assert len(x) == 9
    colors = dict(high_error=RUST, low_error=TEAL)
    fig = plt.figure(figsize=(175 / 25.4, 119 / 25.4))
    fig.text(.028,.951,"a   Can a pilot identify treatment rankings at risk under joint selection?",
             fontsize=9.5, fontweight="bold", color=INK)
    ax = fig.add_axes([.117,.59,.525,.274])
    for field, marker in [("high_error", "o"), ("low_error", "s")]:
        ax.plot(x, mean_by_budget[field], marker=marker, markerfacecolor=colors[field] if field=="high_error" else "white",
                markeredgecolor=colors[field], ms=4.2, lw=1.5, color=colors[field], clip_on=False)
    ax.set(xlabel="Paid assays", ylabel="Wrong treatment signs (%)", xticks=x, ylim=(0,None))
    ax.grid(axis="y", color=LIGHT, lw=.6)
    ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.set_axisbelow(True)
    for side in ("right", "top"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2.2, width=.6, labelsize=7)
    for yy, col, marker, txt in [(.837,RUST,"o","Above median pilot risk"),(.779,TEAL,"s","At or below median risk")]:
        fig.add_artist(plt.Line2D([.680,.714],[yy,yy],color=col,lw=1.4,transform=fig.transFigure))
        fig.add_artist(plt.Line2D([.697],[yy],marker=marker,ms=4,markeredgecolor=col,
                                 markerfacecolor=col if marker=="o" else "white",transform=fig.transFigure))
        fig.text(.726, yy, txt, va="center", fontsize=7.5, color=INK)
    later=sensitivity.loc[("postpilot_noncensus","unresolved_at_pilot")]
    fig.text(.680,.700, "AUC after the pilot", fontsize=8.2, color=INK, weight="bold")
    fig.text(.680,.646, "Pilot model",fontsize=7.8,color=INK)
    fig.text(.901,.646, f"{later.auc_warning_mean:.3f}",fontsize=8.1,color=INK,weight="bold",ha="right")
    fig.text(.680,.596, "Olsen-P gap",fontsize=7.8,color=INK)
    fig.text(.901,.596, f"{later.auc_olsen_mean:.3f}",fontsize=8.1,color=INK,weight="bold",ha="right")
    fig.text(.028,.437,"b   What changes when the warning directs the next assay?",
             fontsize=9.5, fontweight="bold", color=INK)
    fig.text(.028,.389,"Difference in loss, warning minus comparator",fontsize=7.7,color=INK)
    grid=fig.add_gridspec(1,3,left=.220,right=.975,bottom=.128,top=.292,wspace=.30)
    for j,(group,label,mult) in enumerate(OUTCOMES):
        ax=fig.add_subplot(grid[j])
        rivals = sorted(set(means.loc[group].index)-{FOCAL})
        ref = best_design(means, group, rivals)
        lims=[0.]
        for y, key, shade, mark in [(1,ORIGINAL,TEAL,"o"),(0,ref,RUST,"D")]:
            entry=pairs.loc[(group,key)]
            val,lo,hi=entry[["mean","lower","upper"]].to_numpy()*mult
            lims.extend([lo, hi])
            ax.plot([lo,hi],[y,y],lw=1.6,color=shade,solid_capstyle="round")
            ax.scatter([val],[y],marker=mark,s=21,facecolors=shade if y==1 else "white",
                       edgecolors=shade,linewidths=1.1,zorder=3)
        pad=max(max(lims)-min(lims), .005)*.10
        ax.set_xlim(min(lims)-pad,max(lims)+pad)
        ax.axvline(0,color="#6D7F86",lw=.8,ls="--",zorder=0)
        ax.set(ylim=(-.48,1.48), yticks=[1,0],
               yticklabels=["Joint selection","Best alternative"] if j==0 else [],
               xlabel="Percentage points" if mult==100 else "Scaled MSE")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3,min_n_ticks=2))
        ax.tick_params(axis="x",length=2.2,pad=2,labelsize=7)
        ax.tick_params(axis="y",length=0,pad=4,labelsize=7.3)
        ax.set_title(label,loc="left",fontsize=8.3,color=INK,weight="bold",pad=7)
        for side in ("top","right","left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#839098")
    out.mkdir(parents=True,exist_ok=True)
    for suffix in ["pdf","eps","png"]:
        fig.savefig(out / f"FigS9.{suffix}",dpi=360,facecolor="white")
    plt.close(fig)


def start(caption, label, layout, headings):
    return [r"\begin{table}[!htbp]\centering\small",
            r"\caption{"+caption+r"}\label{tab:"+label+"}",
            r"\setlength{\tabcolsep}{3.3pt}\begin{tabular}{@{}"+layout+r"@{}}\toprule",
            " & ".join(headings)+ROW+r"\midrule"]


def finish():
    return r"\bottomrule\end{tabular}\end{table}"


def interval_text(row, multiplier, digits):
    vals=[row[k]*multiplier for k in ["mean","lower","upper"]]
    while any(0 < abs(v) < .5*10**(-digits) for v in vals) and digits < 7:
        digits+=1
    return "$"+f"{vals[0]:.{digits}f}\\;[{vals[1]:.{digits}f},{vals[2]:.{digits}f}]$"


def build_tables(data):
    means,pairs,diagnostic,budgets,sensitivity = data
    # Descriptive, strictly subsequent checks remove the budget at which the warning was issued.
    subsequent=budgets[budgets.budget>budgets.budget.min()]
    additional=[]
    for metric in ["auc","high_error","low_error"]:
        values=subsequent.groupby("pilot")[metric].mean().dropna()
        average=values.mean()
        half=1.95996398454*values.std(ddof=1)/np.sqrt(len(values))
        additional.append(dict(period="after_pilot",metric=metric,mean=average,
                               lower=average-half,upper=average+half,
                               defined_budget_draws=subsequent[metric].count(),total_budget_draws=len(subsequent)))
    diagnostic=pd.concat([diagnostic,pd.DataFrame(additional).set_index(["period","metric"])])
    rows=start("Park Grass pilot treatment-risk discrimination. AUC compares an erroneously signed "
               "and a correctly signed treatment comparison under joint selection at the same pilot "
               "and budget. The high-risk group lies above its pilot median. Rates are percentages among "
               "initially unresolved treatment comparisons with a nonzero reference difference. "
               "Settings count defined budget--pilot values. Non-census summaries average within each "
               "pilot first. The eight later budgets omit the decision point at which the warnings were issued. "
               "The nine-budget primary AUC interval is simultaneous across the 70 primary comparisons. "
               r"Other intervals are pointwise 95\% Monte Carlo intervals. For the later budgets, the final rows "
               "compare the warning AUC with ranking by the smallest absolute pre-existing Olsen-P difference. "
               "The second row restricts ranking performance to comparisons still unresolved at each budget.",
               "alert-forecast","llrr",["Period","Measure","Mean [interval]","Settings"])
    for period,periodname in [("pilot","Paid pilot"),("noncensus","Nine budgets"),("after_pilot","Eight later budgets")]:
        for key,lbl,mul,digits in [("auc","AUC",1,3),("high_error","Above-median error",100,2),
                                    ("low_error","At/below-median error",100,2)]:
            row=diagnostic.loc[(period,key)]
            n=int(row.defined_budget_draws)
            total=int(row.total_budget_draws)
            rows.append(" & ".join([periodname if key=="auc" else "",lbl,interval_text(row,mul,digits),
                                    f"{n}/{total}"])+ROW)
        if period!="after_pilot":
            rows.append(r"\addlinespace")
    rows.extend([r"\midrule",
                 r"\multicolumn{4}{@{}l}{\emph{Discrimination after the pilot}}"+ROW,
                 "Comparison set & Pilot AUC & Olsen-gap AUC & Difference [interval]"+ROW])
    for key,caption in [("unresolved_at_pilot","Initially unresolved"),("still_unresolved","Still unresolved")]:
        row=sensitivity.loc[("postpilot_noncensus",key)]
        diff={k:row["paired_difference_"+k] for k in ["mean","lower","upper"]}
        rows.append(" & ".join([caption,f"{row.auc_warning_mean:.3f}",f"{row.auc_olsen_mean:.3f}",
                                interval_text(diff,1,3)])+ROW)
    rows.append(finish())
    rows.extend(start("Park Grass losses at identical assay budgets. "
                      "Treatment and temporal sign errors are percentages. Combined magnitude error is scaled MSE. "
                      "Each mean weights the nine non-census budgets equally within each of 200 common pilots. "
                      "Bold entries mark the lowest observed loss in each column.",
                      "alert-all","lrrr",["Allocation","Treatment","Time","MSE"]))
    order=[FOCAL,ORIGINAL,"alternate","split_direct","pooled_direct","split_bernoulli",
           *[k for k in NAMES if k not in [ORIGINAL,"alternate"]]]
    assert set(order)==set(means.loc["primary"].index)
    for idx,design in enumerate(order):
        if idx in [1,6]:
            rows.append(r"\midrule")
        numbers=[]
        for group,label,multiplier in OUTCOMES:
            value=means[(group,design)]
            display=f"{value*multiplier:.2f}" if multiplier==100 else f"{value:.3f}"
            if value==means.loc[group].min():
                display=r"\textbf{"+display+"}"
            numbers.append(display)
        rows.append(" & ".join([LABELS[design],*numbers])+ROW)
    rows.append(finish())
    rows.extend(start("Park Grass differences in loss for the pilot-warning allocation minus the named "
                      "comparison method. The lowest-loss rule is selected separately for each outcome "
                      "from the 23 comparators. We also identify the lowest loss among the 18 benchmark rules. "
                      "A shared winner appears once. Sign differences are percentage points. "
                      r"Brackets give simultaneous 95\% Monte Carlo intervals across the 70 primary comparisons.",
                      "alert-paired","lllr",["Outcome","Role","Allocation","Difference [interval]"]))
    allprior=set(order)-{FOCAL}
    legacy=set(NAMES)-{ORIGINAL,"alternate"}
    for group,label,multiplier in OUTCOMES:
        best_initial=best_design(means,group,legacy)
        best_prior=best_design(means,group,allprior)
        refs=[("Joint",ORIGINAL),("Component","contrast_mse" if group=="time" else "sign_error")]
        if best_initial==best_prior:
            refs.append(("Best of 18 and 23",best_prior))
        else:
            refs.extend([("Best of 18",best_initial),("Best of 23",best_prior)])
        for i,(role,design) in enumerate(refs):
            row=pairs.loc[(group,design)]
            rows.append(" & ".join([label if i==0 else "",role,LABELS[design],
                                    interval_text(row,multiplier,2 if multiplier==100 else 3)])+ROW)
        if group!="primary":
            rows.append(r"\addlinespace")
    rows.append(finish())
    return "\n".join(rows)+"\n"


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,default=HERE/"alert_results")
    parser.add_argument("--output",type=Path,default=Path("artwork"))
    parser.add_argument("--tables",type=Path,default=Path("AlertTables.tex"))
    args=parser.parse_args()
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7.8,"axes.linewidth":.6,
                         "axes.edgecolor":"#839098","pdf.fonttype":42,"ps.fonttype":42})
    data=read_results(args.results)
    render_figure(data,args.output)
    args.tables.parent.mkdir(parents=True,exist_ok=True)
    args.tables.write_text(build_tables(data),encoding="utf-8")
