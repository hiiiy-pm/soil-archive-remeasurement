"""Load same-sample bicarbonate P and fusion total P from Hailun's field archive."""
from fractions import Fraction

from trials import Channel, ROOT, discrepancy_kernel, np, original, pd


SITE = "hailun"
SOURCE = ROOT / "data" / "source" / "Hailun" / "paired_soils.csv"


def _data(name):
    if name != SITE:
        raise KeyError(name)
    data = pd.read_csv(SOURCE, dtype=str, keep_default_na=False)
    assert len(data) == 366 and data.source_row.is_unique
    assert not data.duplicated(["year", "plot", "subdivision", "depth_cm"]).any()
    return data


def load_auxiliary_exact(name=SITE):
    data = _data(name)
    return {
        "P_"+depth: group.olsen_p_mg_kg.map(Fraction).to_dict()
        for depth, group in data.groupby("depth_cm", sort=True)
    }


def load_trial(name=SITE):
    data = _data(name)
    frame = pd.DataFrame(dict(year=data.year.astype(int), treatment=data.treatment,
                              depth=data.depth_cm, plot=data["plot"],
                              record_id="Hailun-row-" + data.source_row))
    x = data.olsen_p_mg_kg.astype(float).to_numpy()
    y = data.total_p_mg_kg.astype(float).to_numpy()
    exact_y = data.total_p_mg_kg.map(Fraction).to_dict()
    rows = np.arange(len(data))
    depths = sorted(data.depth_cm.unique())
    panels, responses, fractions = [], {}, {}
    for depth in depths:
        label = "P_"+depth
        mask = frame.depth.eq(depth).to_numpy()
        panels.append(original.engine.make_panel(label, frame, x, mask, "mg/kg", len(depths)))
        responses[label] = y.copy()
        fractions[label] = exact_y
    channel = Channel("P", rows, x, discrepancy_kernel(frame, x), list(range(len(panels))))
    return frame, panels, [channel], responses, fractions
