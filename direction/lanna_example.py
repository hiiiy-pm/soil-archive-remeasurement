"""Reconstruct both complete Lanna field-by-treatment panels, 1980 to 2021."""
import argparse
import json
from fractions import Fraction
from pathlib import Path

from multiperiod_run import OUT
from multiperiod_trial import SITE, load_trial
from trials import np, pd
from experiment import conditional


RULES = ("joint_maximin", "alternate", "sign_error", "contrast_mse")
PHASES = ([0], [1, 2], [3], [4])


def run():
    frame, panels, channels, response, exact = load_trial()
    ch, = channels
    source = OUT / "benchmarks" / SITE
    cfg = json.loads((source / "config.json").read_text())
    n, budgets = len(frame), cfg["budgets"]
    assert cfg["replicates"] == 200 and cfg["ratio"] == 1 and n == 41
    full = frame.groupby("year").apply(
        lambda g: len(g[["field", "treatment"]].drop_duplicates()) == 6,
        include_groups=False)
    assert full[full].index.tolist() == [1980, 2021]
    half = budgets.index(21)
    reference = response[panels[0].name]
    raw = exact[panels[0].name]
    cov = ch.kernel + cfg["ratio"] * np.eye(n)
    matrix, ids, truths = [], [], []
    for field in sorted(frame.field.unique()):
        for treatment in sorted(frame.treatment.unique()):
            member = frame.field.eq(field) & frame.treatment.eq(treatment)
            coeff = np.zeros(n)
            truth = Fraction()
            for year, weight in [(1980, -1), (2021, 1)]:
                rr = np.flatnonzero(member & frame.year.eq(year))
                assert len(rr) == (1 if year == 1980 else 2)
                coeff[rr] = weight / len(rr)
                truth += weight * sum((raw[i] for i in rr), Fraction()) / len(rr)
            matrix.append(coeff)
            truths.append(truth)
            ids.append((field, treatment))
    c = np.array(matrix)
    y = np.array([float(t) for t in truths])
    losses = np.zeros((5, len(budgets) - 1, 6, 2))
    comparator_ids = [cfg["designs"].index(rule) for rule in RULES[2:]]
    for i in range(cfg["replicates"]):
        name = f"draw_{i:03d}.npz"
        with np.load(source / name) as saved, np.load(OUT / "fixed" / SITE / name) as fixed:
            np.testing.assert_array_equal(saved["pilot"], fixed["pilot"])
            base = np.unpackbits(saved["selections"][comparator_ids], axis=-1, count=n)
            paths = fixed["paths"]
            cache = {}
            for bi, b in enumerate(budgets[:-1]):
                masks = [tuple(sorted(path[:b])) for path in paths]
                masks += [tuple(np.flatnonzero(strategy[bi])) for strategy in base]
                for d, mask in enumerate(masks):
                    if mask not in cache:
                        paid = np.array(mask)
                        assert len(np.unique(paid)) == b
                        pred = ch.x + conditional(cov, paid, reference[paid] - ch.x[paid])
                        pred[paid] = reference[paid]
                        estimated = (c @ np.column_stack([pred, pred]))[:, 0]
                        unpaid = np.ones(n, dtype=bool)
                        unpaid[paid] = False
                        resolved = ~c[:, unpaid].any(axis=1)
                        estimated[resolved] = y[resolved]
                        cache[mask] = np.column_stack((np.sign(estimated) != np.sign(y), (estimated-y)**2))
                    losses[d, bi] += cache[mask]
    losses /= cfg["replicates"]
    published = pd.read_csv(OUT / SITE / "year_pairs.csv").query("first_year==1980 and last_year==2021")
    published = published.set_index(["domain", "design"])
    rows = [dict(field=f, treatment=t, first_year=1980, last_year=2021, paid=21, pilots=200,
                 exact_change=str(change), change_mg_kg=float(change)) for (f,t), change in zip(ids,truths)]
    for rule, phase in zip(RULES, PHASES):
        # Average losses from the two starting orders, not their point estimates.
        average = losses[phase].mean(axis=0)
        for field in sorted(frame.field.unique()):
            group = [k for k,(f,t) in enumerate(ids) if f == field]
            values = average[:, group].mean(axis=(0, 1))
            check = published.loc[("P_"+field, rule)]
            np.testing.assert_allclose([values[0], np.sqrt(values[1])],
                                       [check.direction, check.native_rmse], rtol=1e-10, atol=1e-12)
        for row, (wrong, mse) in zip(rows, average[half]):
            row[rule + "_sign_error"] = wrong
            row[rule + "_rmse_mg_kg"] = np.sqrt(mse)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / SITE / "field_example.csv", index=False)
    return result


def format_table(data):
    lines = [r"\begin{table}[!htbp]\centering\small",
             r"\caption{Reference change in acid-extractable soil P from 1980 to 2021 at Lanna and its recovery at 21 of 41 assays. These were the two campaigns sampling all three treatments in both fields. Change and root mean squared error (RMSE) are in mg kg$^{-1}$. Sign error is the percentage of 200 paired pilots. Alternation averages the losses from its two starting orders.}\label{tab:lanna-field-example}",
             r"\setlength{\tabcolsep}{2.8pt}\begin{tabular}{@{}llrrrrrrrrr@{}}\toprule",
             r"&&&\multicolumn{2}{c}{Joint}&\multicolumn{2}{c}{Alternation}&\multicolumn{2}{c}{Direct sign}&\multicolumn{2}{c}{Matched MSE}\\",
             r"Field&Treatment&Change&Sign (\%)&RMSE&Sign (\%)&RMSE&Sign (\%)&RMSE&Sign (\%)&RMSE\\\midrule"]
    for _, r in data.iterrows():
        label = {"Control":"Control", "P":"Annual P", "P6":"Six-year P"}[r.treatment]
        cells = [r.field, label, f"{r.change_mg_kg:.2f}"]
        for rule in RULES:
            cells.extend([f"{100*r[rule+'_sign_error']:.2f}", f"{r[rule+'_rmse_mg_kg']:.2f}"])
        lines.append(" & ".join(cells) + r"\\")
    return "\n".join(lines+[r"\bottomrule\end{tabular}\end{table}"])+"\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path)
    args = parser.parse_args()
    rows = run()
    if args.table:
        args.table.write_text(format_table(rows), encoding="utf-8")
    print(rows.round(4).to_string(index=False))
