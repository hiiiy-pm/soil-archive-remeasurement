"""Load paired plant-available and reference P in the Lanna long-term experiment."""
from fractions import Fraction

from trials import Channel, ROOT, discrepancy_kernel, np, original, pd


SITE = "lanna"
SOURCE = ROOT / "data" / "source" / "Lanna"


def load_trial(name=SITE):
    if name != SITE:
        raise KeyError(name)
    paired = pd.read_csv(SOURCE / "paired_soils.csv", dtype=str, keep_default_na=False)
    latest = paired[paired.year.eq("2021")]
    earlier = paired[paired.year.ne("2021")]
    assert len(paired) == 41 and paired.year.nunique() == 10 and paired.source_line.is_unique
    assert not earlier.duplicated(["year", "field", "treatment"]).any()
    assert len(latest) == 12 and (latest.groupby(["field", "treatment"]).size() == 2).all()
    assert set(earlier[earlier.year.eq("1980")].field) == {"Low", "High"}

    frame = pd.DataFrame(dict(year=paired.year.astype(int), treatment=paired.treatment,
                              field=paired.field, depth="0-20 cm", plot=paired.field,
                              record_id="Time1001-line-" + paired.source_line))
    # The only recurrent spatial identifier is the field half. Using a row
    # identifier as a plot code would imply unsupported physical plot matches.
    rows = np.arange(len(frame))
    x = paired.al_p_mg_kg.map(Fraction).astype(float).to_numpy()
    fractions_by_row = dict(zip(rows, paired.tp_mg_kg.map(Fraction)))
    y = np.array([float(fractions_by_row[i]) for i in rows])
    fields = sorted(frame.field.unique())
    panels, responses, exact = [], {}, {}
    for field in fields:
        label = "P_" + field
        panel = original.engine.make_panel(label, frame, x, frame.field.eq(field).to_numpy(),
                                           "mg/kg", len(fields))
        panels.append(panel)
        responses[label] = y.copy()
        exact[label] = fractions_by_row
    channel = Channel("P", rows, x, discrepancy_kernel(frame, x), list(range(len(fields))))
    return frame, panels, [channel], responses, exact
