"""Compare pilot warnings with an Olsen-P-only ranking under the original joint path."""

import json
from fractions import Fraction
from statistics import NormalDist

from scipy.stats import rankdata

from alert_run import OUT, SITE
from alert_summary import auc, interval
from alert_trial import SOURCE, load_trial
from trials import TargetSet, exact_targets, np, pd


def treatment_inputs():
    """Preserve exact ties between contrasts in the original reported Olsen values."""
    frame, panels, channels, _, _ = load_trial()
    channel, = channels
    panel, = panels
    assert np.array_equal(channel.rows, np.arange(len(frame)))
    archive = pd.read_csv(SOURCE / "paired_soils.csv", dtype=str, keep_default_na=False)
    auxiliary = {panel.name: dict(enumerate(map(Fraction, archive.olsen_p_mg_kg)))}
    targets = TargetSet(panels, channel)
    contrasts = exact_targets(panels, targets, auxiliary)
    treatment = np.concatenate([np.arange(a, b) for _, _, family, a, b, _ in targets.groups
                                if family == "treatment"])
    values = [-abs(contrasts[k]) for k in treatment]
    ordinal = {v: i for i, v in enumerate(sorted(set(values)))}
    # A smaller absolute Olsen-P gap means a larger predicted chance of reversal.
    scores = np.array([ordinal[v] for v in values])
    return scores, targets.C[treatment] != 0


def summarize(group, period, eligibility):
    """Average defined budget comparisons within pilots before pooling pilots."""
    valid = group.dropna(subset=["auc_warning", "auc_olsen"])
    by_pilot = valid.groupby("pilot")[["auc_warning", "auc_olsen"]].mean()
    by_pilot["paired_difference"] = by_pilot.auc_warning - by_pilot.auc_olsen
    row = dict(period=period, eligible_comparisons=eligibility, total_pilot_budgets=len(group),
               defined_pilot_budgets=len(valid), defined_pilots=len(by_pilot),
               eligible_contrast_instances=int(group.eligible.sum()),
               wrong_sign_instances=int(group.events.sum()))
    critical = NormalDist().inv_cdf(.975)
    for col, draws in by_pilot.items():
        estimates = interval(draws.to_numpy(), critical)
        for stat in ("mean", "lower", "upper"):
            row[f"{col}_{stat}"] = estimates[stat]
    return row


def analyze():
    cfg = json.loads((OUT / "benchmarks" / SITE / "config.json").read_text())
    budgets = cfg["budgets"][:-1]
    olsen_rank, support = treatment_inputs()
    observations = []
    for i in range(200):
        name = f"draw_{i:03d}.npz"
        with (np.load(OUT / "fixed" / SITE / name) as fixed,
              np.load(OUT / "warning" / name) as warning):
            path = fixed["paths"][0]
            assert np.array_equal(fixed["pilot"], warning["pilot"])
            position = np.empty(len(path), int)
            position[path] = np.arange(1, len(path) + 1)
            resolved_at = np.where(support, position[None, :], 0).max(axis=1)
            unresolved_at_pilot = resolved_at > budgets[0]
            np.testing.assert_array_equal(unresolved_at_pilot, warning["initially_unresolved"])
            initial = unresolved_at_pilot & warning["reference_nonzero"]
            probabilities = warning["risk"]
            wrong = warning["joint_wrong"]
            for b, err in zip(budgets, wrong):
                assert not err[resolved_at <= b].any()
                for label, eligible in (("unresolved_at_pilot", initial),
                                        ("still_unresolved", initial & (resolved_at > b))):
                    outcomes = err[eligible]
                    score_warning = auc(rankdata(probabilities[eligible]), outcomes)
                    score_olsen = auc(rankdata(olsen_rank[eligible]), outcomes)
                    assert np.isfinite(score_warning) == np.isfinite(score_olsen)
                    observations.append(dict(pilot=i, budget=b, stratum=label,
                                             eligible=int(eligible.sum()), events=int(outcomes.sum()),
                                             auc_warning=score_warning, auc_olsen=score_olsen))

    data = pd.DataFrame(observations)
    rows = []
    for label, g in data.groupby("stratum", sort=False):
        rows.append(summarize(g, "noncensus", label))
        rows.append(summarize(g[g.budget > budgets[0]], "postpilot_noncensus", label))
        for b, group in g.groupby("budget"):
            rows.append(summarize(group, str(b), label))
    destination = OUT / "warning_sensitivity.csv"
    pd.DataFrame(rows).to_csv(destination, index=False)
    print(f"Saved {len(rows)} descriptive comparisons in {destination}")


if __name__ == "__main__":
    analyze()
