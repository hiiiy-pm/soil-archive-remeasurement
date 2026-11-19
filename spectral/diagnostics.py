"""Budget-specific paired comparisons and pilot instability diagnostics."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
comparisons = []
pilot_diagnostics = []
for site, benchmark in [('beltsville', 'concentration'), ('lincoln', 'maximin')]:
    with np.load(ROOT / f'{site}.npz') as archive:
        scores = np.einsum('rdpmfk,f->rdpmk', archive['scores'], archive['joint_weights'])
        names = list(archive['designs'])
        candidate = scores[:, names.index('pooled_retained_contrast'), :, 1]
        benchmarks = [(benchmark, 0), (benchmark, 1), ('retained_contrast', 0),
                      ('pooled_concentration', 1), ('pooled_conversion', 1)]
        for name, model in benchmarks:
            reference = scores[:, names.index(name), :, model]
            periods = [(str(b), [i]) for i, b in enumerate(archive['budgets'])]
            periods.extend([('noncensus', slice(None, -1)), ('after_pilot', slice(1, -1))])
            for period, indices in periods:
                candidate_reps = candidate[:, indices].mean(axis=1)
                reference_reps = reference[:, indices].mean(axis=1)
                differences = candidate_reps - reference_reps
                for k, metric in enumerate(archive['metrics']):
                    paired = differences[:, k]
                    mean = paired.mean()
                    half_width = 1.96 * paired.std(ddof=1) / np.sqrt(len(paired))
                    comparisons.append(dict(
                        site=site, benchmark=name, reconstruction=archive['reconstructions'][model],
                        metric=metric, budget=period, candidate=candidate_reps[:, k].mean(),
                        reference=reference_reps[:, k].mean(), paired_difference=mean,
                        lower=mean-half_width, upper=mean+half_width))
        for nugget in np.unique(archive['nuggets']):
            mask = archive['nuggets'] == nugget
            pilot_diagnostics.append(dict(
                site=site, variance_ratio=nugget, count=mask.sum(),
                curve_scaled_mse=candidate[mask, :-1, 0].mean(),
                first_additional_budget_mse=candidate[mask, 1, 0].mean()))

pd.DataFrame(comparisons).to_csv(ROOT / 'paired_budget_diagnostics.csv', index=False)
pd.DataFrame(pilot_diagnostics).to_csv(ROOT / 'pilot_calibration.csv', index=False)
