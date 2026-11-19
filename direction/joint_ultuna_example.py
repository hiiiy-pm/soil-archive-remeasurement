"""Identical 50-assay Ultuna manure/control contrasts for four allocation rules."""
import json
from fractions import Fraction

from trials import ROOT, load_trial, np, pd
from experiment import conditional


PAIRS = (
    ((1979, 'J'), (1979, 'B')),
    ((1989, 'B'), (1979, 'B')),
    ((1989, 'J'), (1979, 'J')),
    ((1989, 'J'), (1989, 'B')),
)
RULES = ('sign_error', 'contrast_mse', 'joint_maximin', 'alternate')


def run():
    original = ROOT / 'direction/results/ultuna'
    joint = ROOT / 'direction/joint_results'
    config = json.loads((original / 'config.json').read_text())
    frame, panels, channels, references, fractions = load_trial('ultuna')
    budget = 50
    budget_index = config['budgets'].index(budget)
    n_pilots = config['replicates']
    base_indices = [config['designs'].index(k) for k in RULES[:2]]
    selections = []
    for rep in range(n_pilots):
        name = f'draw_{rep:03d}.npz'
        with np.load(original / name) as old, np.load(joint / 'ultuna' / name) as new:
            np.testing.assert_array_equal(old['pilot'], new['pilot'])
            base_masks = np.unpackbits(old['selections'][base_indices, budget_index],
                                       axis=-1, count=len(frame)).astype(bool)
            new_paths = new['paths'][:, :budget]
        record_sets = [np.flatnonzero(mask) for mask in base_masks] + list(new_paths)
        assert all(len(np.unique(ids)) == budget for ids in record_sets)
        selections.append(record_sets)

    published = pd.read_csv(original / 'manure_control.csv', dtype={'exact': str})
    published = published.set_index(['element', 'comparison', 'design'])
    rows = []
    for ch in channels:
        panel = panels[ch.panel_ids[0]]
        records = frame.iloc[ch.rows]
        groups = {group:np.flatnonzero(records.year.eq(group[0]) & records.treatment.eq(group[1]))
                  for pair in PAIRS for group in pair}
        sums = {g:sum((fractions[panel.name][ch.rows[i]] for i in ids), Fraction()) / len(ids)
                for g, ids in groups.items()}
        cov = ch.kernel + config['ratio'] * np.eye(len(ch.rows))
        estimates = np.empty((n_pilots, 5, len(PAIRS)))
        for rep, record_sets in enumerate(selections):
            for path_index, ids in enumerate(record_sets):
                paid = np.flatnonzero(np.isin(ch.rows, ids))
                values = references[panel.name][ch.rows[paid]]
                pred = ch.x + conditional(cov, paid, values - ch.x[paid])
                pred[paid] = values
                estimates[rep, path_index] = [pred[groups[left]].mean() - pred[groups[right]].mean()
                                              for left, right in PAIRS]
        for j, (left, right) in enumerate(PAIRS):
            truth = sums[left] - sums[right]
            comparison = f'{left[0]} {left[1]} minus {right[0]} {right[1]}'
            row = dict(element=ch.name, comparison=comparison, exact=str(truth),
                       truth_mg_kg=float(truth), paid_records=budget, pilots=n_pilots)
            for rule, phases in zip(RULES, ([0], [1], [2], [3, 4])):
                pred = estimates[:, phases, j]
                # Integrate losses over the starting coin within each pilot. Each
                # phase has the same budget; squaring the phase-mean is incorrect.
                mean = pred.mean(axis=1).mean()
                mse = ((pred - float(truth))**2).mean(axis=1).mean()
                wrong = (np.sign(pred) != np.sign(float(truth))).mean(axis=1).mean()
                row[f'{rule}_mean_mg_kg'] = mean
                row[f'{rule}_rmse_mg_kg'] = np.sqrt(mse)
                row[f'{rule}_direction_error'] = wrong
                if rule in RULES[:2]:
                    ref = published.loc[(ch.name, comparison, rule)]
                    assert truth == Fraction(ref['exact'])
                    np.testing.assert_allclose([mean, np.sqrt(mse), wrong],
                        ref[['mean_estimate', 'rmse', 'direction_error']].to_numpy(dtype=float),
                        rtol=1e-12, atol=1e-12)
            rows.append(row)
    table = pd.DataFrame(rows).sort_values(['element', 'comparison'])
    assert len(table) == 8 and n_pilots == 200
    table.to_csv(joint / 'manure_control.csv', index=False)
    print(table.round(4).to_string(index=False))


if __name__ == '__main__':
    run()
