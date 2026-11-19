"""Assemble the Högåsa archive from the R3-9001 public soil measurements."""
from fractions import Fraction

from trials import Channel, ROOT, discrepancy_kernel, np, original, pd


SITE = "hogasa"
SITE_ID = "E-10-1966"
SOURCE = ROOT / "data" / "source" / "Swedish_fertility" / "hogasa_soil.tsv"
VARIABLES = ["AL_P", "HCl_P", "AL_K", "HCl_K"]


def load_trial(name=SITE):
    """Return the same public frame, panels and private score references as trials.load_trial."""
    if name != SITE:
        raise KeyError(name)
    raw = pd.read_csv(SOURCE, sep="\t", dtype=str, keep_default_na=False)
    key = ["year", "component", "plotID", "treatment_code"]
    soil = raw.loc[
        (raw.ID == SITE_ID) & raw.component.str.startswith("soil_")
        & raw.variable.isin(VARIABLES) & ~raw.value.isin(["", "NA"])
    ].copy()
    assert set(soil.unit) == {"mg/kg"}
    # Unlabelled plots cannot identify a physical sample. None has paired HCl at this site.
    assert not soil.loc[soil.plotID.isin(["", "NA", "X"]), "variable"].str.startswith("HCl").any()
    soil = soil[~soil.plotID.isin(["", "NA", "X"])]
    soil["exact"] = [Fraction(v) for v in soil.value]
    observed = soil.groupby(key + ["variable"]).exact.agg(["first", "nunique"])
    # Ignore identical copies. With more than one distinct number the source supplies no
    # unique measurement, so leave that analyte unavailable. Other analytes remain eligible.
    usable = observed.loc[observed["nunique"] == 1, "first"]
    table = usable.unstack("variable")
    present = {a: table[["AL_" + a, "HCl_" + a]].notna().all(axis=1) for a in ["P", "K"]}
    table = table.loc[present["P"] | present["K"]].reset_index()
    table["year"] = table.year.astype(int)
    table["plotID"] = table.plotID.astype(int)
    table.sort_values(["year", "component", "plotID"], inplace=True, ignore_index=True)
    assert not table.duplicated(["year", "component", "plotID"]).any()
    assert table.groupby("plotID").treatment_code.nunique().max() == 1
    frame = pd.DataFrame(dict(
        year=table.year, depth=table.component, plot=table.plotID.astype(str),
        treatment=table.treatment_code,
        record_id=table.year.astype(str) + "-" + table.component + "-" + table.plotID.astype(str)
    ))
    masks = {a: table[["AL_" + a, "HCl_" + a]].notna().all(axis=1).to_numpy() for a in ["P", "K"]}
    assert len(frame) == 146 and masks["P"].sum() == 144 and masks["K"].sum() == 64
    domain_count = sum(frame.loc[mask, "depth"].nunique() for mask in masks.values())
    channels, panels, responses, fractions = [], [], {}, {}
    for element, eligible in masks.items():
        rows = np.flatnonzero(eligible)
        x = table.loc[eligible, "AL_" + element].astype(float).to_numpy()
        exact_map = dict(zip(rows, table.loc[eligible, "HCl_" + element]))
        full_x = np.zeros(len(frame))
        full_x[rows] = x
        reference = np.full(len(frame), np.nan)
        reference[rows] = [float(exact_map[r]) for r in rows]
        panel_ids = []
        for depth in sorted(frame.loc[eligible, "depth"].unique()):
            label = element + "_" + depth
            panel_ids.append(len(panels))
            member = eligible & frame.depth.eq(depth).to_numpy()
            panels.append(original.engine.make_panel(label, frame, full_x, member, "mg/kg", domain_count))
            responses[label] = reference.copy()
            fractions[label] = exact_map
        channels.append(Channel(element, rows, x, discrepancy_kernel(frame.iloc[rows], x), panel_ids))
    return frame, panels, channels, responses, fractions
