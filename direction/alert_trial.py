"""Load all same-record soil Olsen P and total P observations from Park Grass."""
from fractions import Fraction

from trials import Channel, ROOT, discrepancy_kernel, np, original, pd


SITE = "parkgrass"
SOURCE = ROOT / "data" / "source" / "ParkGrass"


def load_trial(name=SITE):
    if name != SITE:
        raise KeyError(name)
    data = pd.read_csv(SOURCE / "paired_soils.csv", dtype=str, keep_default_na=False)
    assert data.groupby("year").size().to_dict() == {"1959": 42, "2002": 90, "2017": 101}
    assert not data.duplicated(["year", "plot"]).any() and data.source_row.is_unique
    frame = pd.DataFrame(dict(
        year=data.year.astype(int), treatment=data.treatment, depth="0-23 cm",
        plot=data["plot"], record_id="ParkGrass-row-" + data.source_row,
    ))
    rows = np.arange(len(frame))
    x = data.olsen_p_mg_kg.astype(float).to_numpy()
    y = data.total_p_mg_kg.astype(float).to_numpy()
    fractions = dict(zip(rows, data.total_p_mg_kg.map(Fraction)))
    name = "P_topsoil"
    panel = original.engine.make_panel(name, frame, x, np.ones(len(rows), bool), "mg/kg", 1)
    channel = Channel("P", rows, x, discrepancy_kernel(frame, x), [0])
    return frame, [panel], [channel], {name: y}, {name: fractions}
