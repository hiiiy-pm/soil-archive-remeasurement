"""Extract same-sample Hailun Olsen P and total P with annual method checks."""
import csv
import hashlib
from collections import Counter
from datetime import date
from decimal import Decimal
from pathlib import Path

from openpyxl import load_workbook


SOURCE = Path(__file__).resolve().parent
TREATMENTS = {
    "HLAZH01": "Mineral fertilizer",
    "HLAFZ01": "Unfertilized",
    "HLAFZ02": "Mineral fertilizer plus straw return",
}
EXPECTED = {2004: 18, 2008: 54, 2009: 54, 2010: 48, 2011: 54, 2012: 54,
            2013: 36, 2015: 48}


def main():
    workbook = SOURCE / "measurements.xlsx"
    assert hashlib.sha256(workbook.read_bytes()).hexdigest() == (
        "65c1071b4a712eea829b709372843a57af90640808c8d2484eaa8991410f83df")
    book = load_workbook(workbook, data_only=True, read_only=True)
    methods = {}
    for year, analyte, method, *_ in book["分析方法记录"].values:
        if isinstance(year, int) and analyte in {"全磷", "有效磷", "速效磷"}:
            component = "total" if analyte == "全磷" else "available"
            assert (year, component) not in methods
            methods[year, component] = method
    iterator = iter(book["土壤养分"])
    header = [cell.value for cell in next(iterator)]
    assert (header[14], header[17]) == ("全磷(P g/kg)", "有效磷(P mg/kg)")
    rows, missing, method_conflict = [], Counter(), Counter()
    for number, cells in enumerate(iterator, start=2):
        record = [cell.value for cell in cells]
        year = record[0]
        if not isinstance(year, int):
            continue
        total, available = record[14], record[17]
        if total is None or available is None:
            missing[year] += 1
            continue
        if "碳酸氢钠浸提" not in methods[year, "available"]:
            method_conflict[year] += 1
            continue
        assert "氢氧化钠碱熔" in methods[year, "total"]
        assert (cells[14].number_format, cells[17].number_format) == ("0.000", "0.0")
        total = Decimal(str(total))
        available = Decimal(str(available))
        assert total == total.quantize(Decimal(".001"))
        assert available == available.quantize(Decimal(".1"))
        plot = record[3][:7]
        rows.append(dict(source_row=number, year=year,
                         collection_date=date(year, record[1], record[2]).isoformat(),
                         source_plot=record[3], plot=plot, treatment=TREATMENTS[plot],
                         subdivision=record[5], depth_cm=record[10], source_sample=record[11],
                         olsen_p_mg_kg=format(available, "f"),
                         total_p_mg_kg=format(total*1000, "f")))
    assert missing == {2005: 54, 2006: 18, 2014: 54}
    assert method_conflict == {2007: 54}
    assert Counter(row["year"] for row in rows) == EXPECTED
    assert len({(r["year"], r["plot"], r["subdivision"], r["depth_cm"]) for r in rows}) == len(rows)
    assert len({(r["year"], r["plot"], r["source_sample"]) for r in rows}) == len(rows)
    with (SOURCE/"paired_soils.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
