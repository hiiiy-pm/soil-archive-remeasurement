"""Render three-trial family-separated allocation and ablation results."""
import argparse
from pathlib import Path

import pandas as pd

from joint_tables import NAMES
from ranking_summary import PRIMARY, SITES, FOCAL, ORIGINAL_JOINT, NEW, OUT


LABEL = dict(NAMES,split_direct="Separate sign risks",pooled_direct="Pooled sign risk",
             split_bernoulli="Separate sign variances",joint_maximin="Pooled sign variance")
TRIAL = {"tcse":"Lincoln-TCSE","ultuna":"Ultuna","lanna":"Lanna"}
ROW = r"\\"
END = r"\bottomrule\end{tabular}\end{table}"


def start(caption,label,cols):
    return [r"\begin{table}[!htbp]\centering\small",
            r"\caption{" + caption + r"}\label{tab:" + label + "}",
            r"\setlength{\tabcolsep}{2.2pt}\begin{tabular}{@{}" + cols + r"@{}}\toprule"]


def loss(value,site,metric):
    if metric=="direction":
        return f"{100*value:.2f}"
    return f"{value:.5f}" if site=="tcse" else f"{value:.3f}"


def change(value,site,metric):
    scaled=100*value if metric=="direction" else value
    digits=2 if metric=="direction" else 5 if site=="tcse" else 3
    while scaled and round(scaled,digits)==0:
        digits+=1
    return f"{scaled:.{digits}f}" if scaled else "0"


def limits(row,site,metric):
    value,lo,hi=[change(row[k],site,metric) for k in ["difference","simultaneous_lower","simultaneous_upper"]]
    return "$"+f"{value}\\;[{lo},{hi}]$"


def build(folder):
    raw=pd.read_csv(folder/"performance.csv",dtype={"budget":str})
    primary=raw[(raw.budget=="noncensus") & (raw.model=="kriging")]
    means=primary.set_index(["site","design","group","metric"])["mean"]
    pairs=pd.read_csv(folder/"paired.csv",dtype={"budget":str}).dropna(subset=["simultaneous_upper"])
    pairs=pairs.set_index(["site","focal","comparator","group","metric"])
    best=pd.read_csv(folder/"best_baselines.csv").set_index(["site","group","metric","scope"])
    factorial=pd.read_csv(folder/"factorial.csv").set_index(["site","effect","group","metric"])
    assert len(pairs)==216 and len(factorial)==45
    allocation=start(r"All 23 allocations under common kriging. Time and treatment columns give sign error in percent. "
                     r"MSE gives the scaled mean squared error across both contrast families. "
                     r"The same 200 pilots and nine non-census budgets apply to all rows within each trial. "
                     r"The first four rules all include a magnitude objective.", "ranks-all","lrrrrrrrrr")
    allocation += [r"&\multicolumn{3}{c}{Lincoln-TCSE}&\multicolumn{3}{c}{Ultuna}&\multicolumn{3}{c}{Lanna}\\",
                   r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
                   r"Allocation & Time & Treat. & MSE & Time & Treat. & MSE & Time & Treat. & MSE\\\midrule"]
    order=NEW+[ORIGINAL_JOINT]+[d for d in NAMES if d!=ORIGINAL_JOINT]
    for i,design in enumerate(order):
        if i==4:
            allocation.append(r"\midrule")
        vals=[loss(means.loc[(s,design,g,m)],s,m) for s in SITES for g,m in PRIMARY]
        allocation.append(" & ".join([LABEL[design],*vals])+ROW)
    allocation.append(END)
    comparison=start(r"Family-separated direct sign risks with magnitude selection minus the named allocations. Intervals "
                     r"are simultaneous 95\% Monte Carlo intervals across all 216 comparisons. Negative differences favor "
                     r"the family-separated direct rule. Sign errors are percentage points. The joint rule pools the sign variances "
                     r"across treatment and time comparisons. The component references are matched MSE for time and direct sign for "
                     r"magnitude. Direct sign is also reported for treatment ranking.",
                     "ranks-primary","llrr")
    comparison.append(r"Trial & Outcome & Versus pooled variance & Versus component\\\midrule")
    for i,site in enumerate(SITES):
        for j,((g,m),name,reference) in enumerate(zip(PRIMARY,["Time","Treatment","Magnitude"],["contrast_mse","sign_error","sign_error"])):
            r1=pairs.loc[(site,FOCAL,ORIGINAL_JOINT,g,m)]
            r2=pairs.loc[(site,FOCAL,reference,g,m)]
            comparison.append(" & ".join([TRIAL[site] if j==0 else "",name,
                                           limits(r1,site,m),limits(r2,site,m)])+ROW)
        if i<2:
            comparison.append(r"\addlinespace")
    comparison.append(END)
    benchmarks=start(r"Family-separated direct sign risks with magnitude selection versus the lowest-loss rule among "
                     r"18 allocations for each trial and outcome. The comparator is the same across the nine non-census budgets. "
                     r"Negative differences favor family-separated risks. Brackets give simultaneous 95\% Monte Carlo intervals "
                     r"from the 216-comparison family. Sign losses are in percentage points.",
                     "ranks-best","lllr")
    benchmarks.append(r"Trial & Outcome & Lowest-loss benchmark & Difference [interval]\\\midrule")
    for i,site in enumerate(SITES):
        for j,((g,m),name) in enumerate(zip(PRIMARY,["Time","Treatment","Magnitude"])):
            strongest=best.loc[(site,g,m,"original18"),"comparator"]
            change_row=pairs.loc[(site,FOCAL,strongest,g,m)]
            benchmarks.append(" & ".join([TRIAL[site] if j==0 else "",name,LABEL[strongest],
                                          limits(change_row,site,m)])+ROW)
        if i<2:
            benchmarks.append(r"\addlinespace")
    benchmarks.append(END)
    ablation=start(r"Controlled changes in allocation under the same kriging and magnitude objective. Split minus pooled "
                   r"changes only whether time and treatment sign gains are normalized separately. Risk minus variance "
                   r"changes only the sign acquisition criterion. Each difference is followed by its 95\% interval, adjusted "
                   r"across the same 216 comparisons. Sign losses are in percentage points. Full by-budget and per-domain "
                   r"results and the paired difference in simple effects are available in Online Resource~2.",
                   "ranks-factors","lrrr")
    ablation.append(r"Change & Temporal signs & Treatment signs & Scaled MSE\\\midrule")
    effects=[("split_under_direct","Split $-$ pooled, risk"),("split_under_bernoulli","Split $-$ pooled, variance"),
             ("direct_under_split","Risk $-$ variance, split"),("direct_under_pooling","Risk $-$ variance, pooled")]
    for i,site in enumerate(SITES):
        ablation.append(r"\multicolumn{4}{@{}l}{\emph{"+TRIAL[site]+r"}}"+ROW)
        for j,(effect,label) in enumerate(effects):
            vals=[limits(factorial.loc[(site,effect,g,m)],site,m) for g,m in PRIMARY]
            ablation.append(" & ".join([label,*vals])+ROW)
        if i<2:
            ablation.append(r"\addlinespace")
    ablation.append(END)
    return "\n".join(allocation+comparison+benchmarks+ablation)+"\n"


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,default=OUT)
    parser.add_argument("--output",type=Path,default=Path("RankingTables.tex"))
    args=parser.parse_args()
    args.output.write_text(build(args.results),encoding="utf8")
