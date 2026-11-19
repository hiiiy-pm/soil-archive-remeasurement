"""Prepare the Kelson soil-water test, then release references against the frozen files.

Run 'python prepare.py freeze' and then 'python prepare.py references' in this folder.
For reproduction outside the distributed folder supply --output to both commands.
"""
import argparse
import csv
from collections import defaultdict
from datetime import date
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
from scipy.stats import t

HERE = Path(__file__).resolve().parent
INPUTS = {
    "source/sw.csv": "89ab73eabcda3319d182ead244c9b5408dcce6b45949bcdec5b096c9f072c193",
    "source/TableSX_CZ17O_TripleO_iso_stds_reps.xlsx":
        "e2fd19998ecf7ccbbb56c10844c756bb6748dcb3722b73cf90a6be0ac6889124",
}
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
FT = {"MOJ": "test", "REY": "test", "JOR": "calibration", "ESGR": "calibration"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def store_csv(path, records):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def irms_runs(ids):
    """Read the released workbook, selecting reference values only for named specimens."""
    with ZipFile(HERE / next(k for k in INPUTS if k.endswith(".xlsx"))) as book:
        strings = ["".join(n.text or "" for n in entry.iter(NS + "t"))
                   for entry in ET.fromstring(book.read("xl/sharedStrings.xml"))]
        sheet = ET.fromstring(book.read("xl/worksheets/sheet1.xml"))
    def decode(cell):
        v = cell.findtext(NS + "v", default="")
        return strings[int(v)] if cell.get("t") == "s" else v
    allrows = iter(sheet.iter(NS + "row"))
    headers = {c.attrib["r"].rstrip("0123456789"): decode(c) for c in next(allrows)}
    found = defaultdict(list)
    for row in allrows:
        cells = {headers[c.attrib["r"].rstrip("0123456789")]: c for c in row}
        key = decode(cells["sample.ID"])
        if key not in ids or decode(cells["flag"]) != "0":
            continue
        found[ids[key]].append((int(decode(cells["IPL.num"])),
                                float(decode(cells["D17O.SMOWSLAP.per.meg"]))))
    return {key: sorted(values) for key, values in found.items()}


def inventory():
    for name, expected in INPUTS.items():
        if digest(HERE / name) != expected:
            raise ValueError(f"Source checksum mismatch for {name}")
    with (HERE / "source/sw.csv").open(encoding="utf-8-sig", newline="") as f:
        records = [a for a in csv.DictReader(f) if a["N.IRMS"] not in ("NA", "")
                   and a["d2H.mean"] not in ("NA", "") and a["d18O.mean"] not in ("NA", "")]
    assert len(records) == len({a["Identifier_1"] for a in records}) == len({a["SampleIDLong"] for a in records}) == 36
    return records


def role_data(records, role):
    selected = [a for a in records if FT[a["siteID.1"]] == role]
    aliases = {a[k]: a["Identifier_1"] for a in selected for k in ("Identifier_1", "SampleIDLong")}
    pairs = irms_runs(aliases)
    result = []
    for a in selected:
        runs = pairs[a["Identifier_1"]]
        assert len(runs) == int(a["N.IRMS"])
        reference = np.mean([value for _, value in runs])
        assert math.isclose(reference, float(a["D17O_pmg"]), abs_tol=1e-8)
        result.append((a, runs, float(reference)))
    return result


def auxiliary(a):
    return float(a["d2H.mean"]) - 8 * float(a["d18O.mean"])


def freeze(out):
    records = inventory()
    calibrated = role_data(records, "calibration")
    repeated = [runs for _, runs, _ in calibrated if len(runs) == 2]
    assert len(calibrated) == 11 and len(repeated) == 10
    residual_df = len(repeated)
    sd = float(np.sqrt(sum((r[1][1] - r[0][1])**2 for r in repeated) / (2 * residual_df)))
    x = np.array([auxiliary(a) for a, _, _ in calibrated])
    y = np.array([ref for _, _, ref in calibrated])
    coef = np.linalg.lstsq(np.column_stack((np.ones(len(x)), x)), y, rcond=None)[0]
    lookup = dict(zip(("intercept", "slope"), map(float, coef)))
    table = [dict(sample=a["Identifier_1"], region=a["siteID.1"], d_excess=auxiliary(a),
                  run1=runs[0][0], irms1_per_meg=runs[0][1],
                  run2=runs[1][0] if len(runs) == 2 else "",
                  irms2_per_meg=runs[1][1] if len(runs) == 2 else "", reference=ref)
             for a, runs, ref in calibrated]
    store_csv(out / "calibration.csv", table)
    frame = []
    for a in sorted(records, key=lambda a: int(a["Identifier_1"][2:])):
        if FT[a["siteID.1"]] != "test":
            continue
        assert int(a["N.IRMS"]) == 2
        bottom = float(a["bottom.depth"])
        day = date.fromisoformat(a["date.collection"])
        season = (day.month % 12) // 3 + 1
        yr = day.year + int(day.month == 12)
        sample = a["Identifier_1"]
        frame.append(dict(sample=sample, stratum="shallow" if bottom <= 30 else "deep",
                          group=a["siteID.1"] + "_" + a["siteID.2"], plot=sample,
                          campaign=yr * 10 + season, auxiliary=float(coef[0] + coef[1]*auxiliary(a)),
                          bottom_depth_cm=bottom, collection_date=day.isoformat(),
                          region=a["siteID.1"], source_specimen=a["SampleIDLong"], d_excess=auxiliary(a)))
    assert len(frame) == 25
    store_csv(out / "frame.csv", frame)
    counts = defaultdict(lambda: defaultdict(int))
    for a in frame:
        counts[(a["stratum"], a["group"])][a["campaign"]] += 1
    cutoffs = []
    quantile = float(t.ppf(.975, residual_df))
    for (h, g), cells in sorted(counts.items()):
        for early, late in combinations(sorted(cells), 2):
            n1, n2 = cells[early], cells[late]
            cutoff = quantile * sd * math.sqrt(1/(2*n1) + 1/(2*n2))
            cutoffs.append(dict(stratum=h, group=g, early=early, late=late, limit=cutoff,
                                n_early=n1, n_late=n2, different_season=int(early%10 != late%10)))
    store_csv(out / "limits.csv", cutoffs)
    files = dict(INPUTS)
    for name in ("design.md", "protocol.md", "prepare.py", "evaluate.py"):
        files[name] = digest(HERE / name)
    for name in ("frame.csv", "limits.csv", "calibration.csv"):
        files[name] = digest(out / name)
    meta = dict(unit="per meg", files=files, single_irms_sd=sd, df=residual_df,
                t_multiplier=quantile, calibration_ols=lookup, n_sites_calibration=2,
                calibration_specimens=11, validation_specimens=25, temporal_pairs=len(cutoffs),
                cross_season_pairs=sum(c["different_season"] for c in cutoffs))
    text = json.dumps(meta, indent=2, sort_keys=True) + "\n"
    frozen = out / "lock.json"
    if frozen.exists() and json.loads(frozen.read_text("utf-8")) != meta:
        raise RuntimeError("Existing lock differs. Use a separate --output directory to reproduce.")
    frozen.write_text(text, encoding="utf-8", newline="\n")
    print(f"Calibration and limits frozen for {len(frame)} specimens and {len(cutoffs)} time contrasts.")


def release(out):
    frozen = json.loads((out / "lock.json").read_text("utf-8-sig"))
    for name, expected in frozen["files"].items():
        folder = out if name in ("frame.csv", "limits.csv", "calibration.csv") else HERE
        assert digest(folder / name) == expected, name
    values = {a["Identifier_1"]: ref for a, _, ref in role_data(inventory(), "test")}
    with (out / "frame.csv").open(encoding="utf-8") as f:
        data = [dict(sample=a["sample"], reference=values[a["sample"]]) for a in csv.DictReader(f)]
    assert len(data) == len(values) == frozen["validation_specimens"]
    store_csv(out / "references.csv", data)
    print("Released the held-out duplicate means after verifying the lock.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "references"))
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (freeze if args.phase == "freeze" else release)(args.output)

