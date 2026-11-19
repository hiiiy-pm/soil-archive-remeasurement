"""Describe agreement between recorded Olsen P and total P field contrasts."""
import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path

from olsen_run import auxiliary_exact, loader

HERE = Path(__file__).resolve().parent
import pandas as pd


def tabulate(site):
    field, panels, _, _, references = loader(site)(site)
    auxiliary = auxiliary_exact(site, panels)
    olsen = {i: auxiliary[p.name][i] for p in panels for i in p.rows}
    total = {i: references[p.name][i] for p in panels for i in p.rows}
    output, summary = [], []
    for panel in panels:
        depth = field.iloc[panel.rows]
        means = {}
        for k, group in depth.groupby(["year","treatment"]):
            ids = list(group.index)
            means[k] = tuple(sum(raw[i] for i in ids)/len(ids) for raw in (olsen,total))
        pairs = {
            "treatment": [(a,b) for a,b in combinations(sorted(means),2) if a[0]==b[0]],
            "time": [(b,a) for a,b in combinations(sorted(means),2) if a[1]==b[1]]}
        for family, comparisons in pairs.items():
            counts = Counter()
            for left,right in comparisons:
                d_x,d_y = [means[left][i]-means[right][i] for i in (0,1)]
                if d_x==0 and d_y==0:
                    relation = "both_tied"
                elif d_x==0:
                    relation = "olsen_only_tied"
                elif d_y==0:
                    relation = "total_only_tied"
                else:
                    relation = "agree" if d_x*d_y>0 else "opposite"
                counts[relation] += 1
                output.append(dict(site=site,domain=panel.name,family=family,unit="mg P/kg",
                                   year_left=left[0],group_left=left[1],year_right=right[0],group_right=right[1],
                                   olsen_left_exact=str(means[left][0]),olsen_right_exact=str(means[right][0]),
                                   total_left_exact=str(means[left][1]),total_right_exact=str(means[right][1]),
                                   olsen_gap_exact=str(d_x),total_gap_exact=str(d_y),relation=relation))
            summary.append(dict(site=site,domain=panel.name,family=family,contrasts=len(comparisons),
                                **{k:counts[k] for k in ["agree","opposite","olsen_only_tied","total_only_tied","both_tied"]}))
    return pd.DataFrame(output),pd.DataFrame(summary)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=HERE/"olsen_results")
    args=parser.parse_args()
    contrasts,counts = zip(*(tabulate(site) for site in ("parkgrass","hailun")))
    args.output_dir.mkdir(exist_ok=True,parents=True)
    pd.concat(contrasts,ignore_index=True).to_csv(args.output_dir/"chemistry_contrasts.csv",index=False)
    combined = pd.concat(counts,ignore_index=True)
    combined.to_csv(args.output_dir/"chemistry_concordance.csv",index=False)
    print(combined.to_string(index=False))
