"""Verify the original deposit and regenerate the paired 0–20 cm soil frame."""
import csv
from decimal import Decimal
from hashlib import sha256
from io import BytesIO, StringIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile


ARCHIVE = "https://zenodo.org/api/records/8070747/files/MPIBGC-TEE/Lanna-v1.0.0.zip/content"
ARCHIVE_SHA = "278b110f836a5c0f967dbb8ee9c80de2336ac1a5021da258e32a36c40c126be1"
TABLES = {
    "Time1001.csv": "b2bd31fe8e1864cc50a35dadbe5204462ef775b678bd4137c1d4475c0cbb75c4",
    "R1001.csv": "eb2c503b4cd69fb092cc3db3ad46083381eef75099d21dec22b419821fecf7db",
}


def records(data):
    table = data.decode("utf-8")
    return list(csv.DictReader(table.splitlines(), delimiter=";"))


def number(value):
    return Decimal(value.replace(",", "."))


def prepare(tables):
    source = records(tables["Time1001.csv"])
    companion = records(tables["R1001.csv"])
    latest = [r for r in source if r["Year"] == "2021"]
    assert len(source) == 42 and len(latest) == len(companion) == 12
    for a, b in zip(latest, companion):
        assert (a["Field"], a["Treatment"]) == (b["Field"], b["Treatment"])
        assert number(a["TP"]) == number(b["TP"])
        assert 10 * number(a["PAL"]) == number(b["PAL"])
    out = StringIO(newline="")
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["source_line", "year", "field", "treatment", "al_p_mg_kg", "tp_mg_kg"])
    excluded = []
    for line, r in enumerate(source, start=2):
        if r["TP"] in ("", "nd") or r["PAL"] in ("", "nd"):
            excluded.append((r["Year"], r["Field"], r["Treatment"]))
            continue
        writer.writerow([line, r["Year"], r["Field"], r["Treatment"],
                         10 * number(r["PAL"]), number(r["TP"])])
    assert excluded == [("1968", "Low", "P6")]
    return out.getvalue().encode("utf-8")


def fetch():
    with urlopen(ARCHIVE, timeout=60) as response:
        payload = response.read()
    if sha256(payload).hexdigest() != ARCHIVE_SHA:
        raise ValueError("The downloaded archive differs from the cited deposit.")
    tables = {}
    with ZipFile(BytesIO(payload)) as archive:
        for name, expected in TABLES.items():
            data = archive.read("MPIBGC-TEE-Lanna-0394941/data/" + name)
            if sha256(data).hexdigest() != expected:
                raise ValueError("Unexpected content for " + name)
            tables[name] = data
            print(name, expected)
    target = Path(__file__).resolve().parent / "paired_soils.csv"
    result = prepare(tables)
    if not target.exists() or target.read_bytes() != result:
        temp = target.with_suffix(".part")
        temp.write_bytes(result)
        temp.replace(target)
    print(target.name, sha256(result).hexdigest())


if __name__ == "__main__":
    fetch()
