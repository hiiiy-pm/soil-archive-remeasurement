"""Extract paired Olsen P and total P measurements from the Park Grass workbook."""
import csv
import hashlib
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

from openpyxl import load_workbook


SOURCE = Path(__file__).resolve().parent
URL = "https://www.era.rothamsted.ac.uk/metadata/rpg5/PGsoilPex/01-PGsoilPex.xlsx"
SHA256 = "43d2276a819381de81e9a3e4ec98059fac8bc4b8d0abe457fa3c9cc94fdc96de"


def main():
    path = SOURCE / "measurements.xlsx"
    if not path.exists():
        with urlopen(URL, timeout=90) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != SHA256:
            raise ValueError("The downloaded workbook differs from the archived deposit.")
        path.write_bytes(payload)
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError("Unexpected source workbook checksum.")
    workbook = load_workbook(path, read_only=True, data_only=True)
    source = iter(workbook["soil_data"].values)
    header = next(source)
    output = []
    for row_number, values in enumerate(source, start=2):
        row = dict(zip(header, values))
        if not all(isinstance(row[name], (int, float)) for name in ("Olsen_P", "Total_P")):
            continue
        output.append({
            "source_row": row_number,
            "year": row["harvest_year"],
            "plot": str(row["plot_id"]).strip(),
            "treatment": str(row["treatment"]).strip(),
            "liming": str(row["liming_factor_level"]).strip(),
            "olsen_p_mg_kg": row["Olsen_P"],
            "total_p_mg_kg": row["Total_P"],
        })
    workbook.close()
    assert Counter(r["year"] for r in output) == {1959: 42, 2002: 90, 2017: 101}
    assert len({(r["year"], r["plot"]) for r in output}) == len(output)
    with (SOURCE / "paired_soils.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=output[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)


if __name__ == "__main__":
    main()
