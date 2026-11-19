"""Load measured Olsen and Truog P on individual Winchmore plot-date-depth samples."""
from fractions import Fraction

from trials import Channel, ROOT, discrepancy_kernel, np, original, pd


SITE = "winchmore"
SOURCE = ROOT/"data/source/Winchmore/paired_soils.csv"


def _read(name):
    if name != SITE:
        raise KeyError(name)
    data = pd.read_csv(SOURCE, dtype=str, keep_default_na=False)
    assert len(data) == 399 and not data.duplicated(["date", "plot", "depth_cm"]).any()
    return data


def load_auxiliary_exact(name=SITE):
    data = _read(name)
    return {"P_"+depth: group.olsen_p_mg_kg.map(Fraction).to_dict()
            for depth, group in data.groupby("depth_cm")}


def load_trial(name=SITE):
    data = _read(name)
    frame = pd.DataFrame(dict(year=data.campaign.astype(int), treatment=data.treatment,
                              depth=data.depth_cm, plot=data["plot"], record_id="Winchmore-row-"+data.source_row))
    x = data.olsen_p_mg_kg.astype(float).to_numpy()
    y = data.truog_p_mg_kg.astype(float).to_numpy()
    exact = data.truog_p_mg_kg.map(Fraction).to_dict()
    depths = sorted(data.depth_cm.unique(), key=lambda field: float(field.split("-")[0]))
    panels, responses, exacts = [], {}, {}
    for depth in depths:
        name = "P_"+depth
        mask = frame.depth.eq(depth).to_numpy()
        panels.append(original.engine.make_panel(name, frame, x, mask, "mg/kg", len(depths)))
        responses[name] = y.copy()
        exacts[name] = exact
    rows = np.arange(len(frame))
    channel = Channel("P", rows, x, discrepancy_kernel(frame, x), list(range(len(panels))))
    return frame, panels, [channel], responses, exacts
