"""Recover observed same-sample Olsen and Truog P from the fertiliser trial."""
import csv
from collections import Counter
from hashlib import sha256
from pathlib import Path

from openpyxl import load_workbook


HERE = Path(__file__).resolve().parent
DEPTH = {"0 - 75": "0-7.5", "75 - 150": "7.5-15", "150 - 225": "15-22.5", "225 - 300": "22.5-30"}


def prepare():
    source = HERE / "measurements.xlsx"
    assert sha256(source.read_bytes()).hexdigest() == "a6f97196ac06725b9341e626038ae835ff2e5e681804b54bb7546acabf3a8199"
    workbook = load_workbook(source, read_only=True, data_only=False)
    records, seen = [], {}
    formula_count = 0
    duplicated = []
    try:
        sheet = workbook["Soil chemistry data"]
        names = next(sheet.values)
        positions = {name: names.index(name) for name in ["Trial ID","BatchID","Sampling date",
                     "Plot","Sampling depth","Treatment","Olsen P","Truog P"]}
        for row in sheet.iter_rows(min_row=2):
            def field(key):
                return row[positions[key]].value

            olsen = row[positions["Olsen P"]]
            truog = field("Truog P")
            if not isinstance(truog, (int, float)):
                continue
            if olsen.data_type == "f":
                assert olsen.font.color.type == "rgb" and olsen.font.color.rgb == "FFFF0000"
                formula_count += 1
                continue
            assert isinstance(olsen.value, (int, float)) and olsen.font.color is None
            date = field("Sampling date")
            item = dict(source_row=olsen.row, trial_code=field("Trial ID"), batch=field("BatchID"),
                        date=date.date().isoformat(), campaign=int(date.strftime("%Y%m%d")),
                        plot=str(field("Plot")), depth_cm=DEPTH[field("Sampling depth")],
                        treatment=field("Treatment"), olsen_p_mg_kg=olsen.value, truog_p_mg_kg=truog)
            key = (item["campaign"], item["plot"], item["depth_cm"])
            if key in seen:
                prior = seen[key]
                assert all(prior[k] == v for k, v in item.items() if k != "source_row")
                duplicated.append((prior["source_row"], item["source_row"]))
            else:
                seen[key] = item
                records.append(item)
    finally:
        workbook.close()
    assert len(records) == 399 and formula_count == 1425 and duplicated == [(1683,1690)]
    assert Counter(r["depth_cm"] for r in records) == {"0-7.5":340,"7.5-15":20,"15-22.5":19,"22.5-30":20}
    assert len({r["campaign"] for r in records if r["depth_cm"] == "0-7.5"}) == 17
    with (HERE/"paired_soils.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    prepare()
