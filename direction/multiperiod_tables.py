"""Build tables for the fixed-rule replication at Lanna."""
import argparse
from pathlib import Path

import pandas as pd
from joint_tables import NAMES, ENDPOINTS
from multiperiod_summary import OUT


FIXED = ["joint_maximin", "alternate"]
ROW = r"\\"
END = r"\bottomrule\end{tabular}\end{table}"


def table(caption, label, columns, headings):
    return [r"\begin{table}[!htbp]\centering\small",
            "\\caption{" + caption + "}\\label{" + label + "}",
            r"\setlength{\tabcolsep}{3.4pt}\begin{tabular}{@{}" + columns + r"@{}}\toprule",
            headings + ROW + r"\midrule"]


def generate(folder):
    def rows(name):
        df = pd.read_csv(folder/(name+".csv"),dtype={"budget":str})
        return df[(df.model=="kriging") & (df.budget=="noncensus")]
    perf = rows("performance").set_index(["design","group","metric"])["mean"]
    paired = rows("paired").set_index(["focal","comparator","group","metric"])
    best = rows("best_baselines").set_index(["focal","group","metric"]).best_original
    outcomes = [*ENDPOINTS, ("primary","small_direction"),("primary","three_class")]
    names = list(NAMES)
    assert len(names)==20 and perf.index.is_unique
    body = table("All fixed allocations at Lanna under common kriging, averaged over nine non-census budgets and 200 pilots. "
                 "Time and treatment sign errors include every nonzero comparison within field. Small differences are the "
                 "lowest quarter within each field and family. Three-class error scores ties as a separate sign. Error rates are percentages.",
                 "tab:lanna-all","lrrrrr","Allocation & Time signs & Treatment signs & Scaled MSE & Small signs & Three-class")
    for rule in names:
        vals=[perf.loc[(rule,g,m)] for g,m in outcomes]
        nums=[f"{v:.4f}" if j==2 else f"{100*v:.2f}" for j,v in enumerate(vals)]
        body.append(" & ".join([NAMES[rule],*nums])+ROW)
    body.append(END)
    body.extend(table("Differences in loss at Lanna for joint or alternating selection minus the named comparator. "
                      "Each cell reports the mean paired difference and its simultaneous 95\\% Monte Carlo interval "
                      "over 111 distinct comparisons. Direction uses percentage points. Negative values favor the column's allocation. "
                      "The best benchmark is selected separately for each outcome among 18 comparators.",
                      "tab:lanna-pairs","llrr","Outcome & Comparator & Joint selection & Alternation"))
    labels=["Temporal signs","Treatment signs","Scaled MSE"]
    for j,(g,m) in enumerate(ENDPOINTS):
        chosen=best.loc[(FIXED[0],g,m)]
        comp=list(dict.fromkeys([["contrast_mse","sign_error","sign_error"][j],"contrast_mse",chosen,"alternate"]))
        for k,other in enumerate(comp):
            line=[labels[j] if k==0 else "",NAMES[other]]
            for focal in FIXED:
                if focal==other:
                    line.append(r"\textemdash")
                    continue
                r=paired.loc[(focal,other,g,m)]
                d,lo,hi=[(1 if j==2 else 100)*r[a] for a in ["difference","simultaneous_lower","simultaneous_upper"]]
                precision=3 if j==2 else 2
                line.append(f"\u0024{d:.{precision}f}\\;[{lo:.{precision}f},{hi:.{precision}f}]\u0024")
            body.append(" & ".join(line)+ROW)
        if j<2: body.append(r"\addlinespace")
    body.append(END)
    byfield=rows("domains").set_index(["domain","family","design"]).native_rmse
    body.extend(table("Root mean squared errors of field-specific treatment and time differences in the source-defined "
                      "reference phosphorus concentration (mg kg$^{-1}$). Squared errors are averaged over all eligible "
                      "differences, nine non-census budgets and 200 pilots before taking the square root.",
                      "tab:lanna-native","llrrrr","Field & Difference & Joint & Alternating & Direct sign & Matched MSE"))
    for domain in sorted(byfield.index.get_level_values("domain").unique()):
        for family in ["time","treatment"]:
            line=[domain.split("_")[-1],family.capitalize(),*[f"{byfield.loc[(domain,family,r)]:.2f}" for r in [*FIXED,"sign_error","contrast_mse"]]]
            body.append(" & ".join(line)+ROW)
    body.append(END)
    return "\n".join(body)+"\n"


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,default=OUT/"lanna")
    parser.add_argument("--output",type=Path,default=Path("MulticampaignTables.tex"))
    args=parser.parse_args()
    args.output.write_text(generate(args.results),encoding="utf8")
