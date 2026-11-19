"""Recover the Högåsa soil excerpt from the unmodified public SND deposit."""
import argparse
import hashlib
import io
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile


URL = "https://api.researchdata.se/dataset/2024-462/1/file/zip"
SOURCE_SHA256 = "9a72ec5d61ac71ba8d2b68c7a5d6582aa40d734415a6b53135dfad3334322c4c"
ROOT = Path(__file__).resolve().parent
VARIABLES = {b"AL_P", b"HCl_P", b"AL_K", b"HCl_K"}


def extract(package, destination):
    if hashlib.sha256(package).hexdigest() != SOURCE_SHA256:
        raise ValueError("Original deposit checksum mismatch")
    with ZipFile(io.BytesIO(package)) as archive:
        checksums = dict(reversed(line.split()) for line in archive.read("sha256.txt").decode().splitlines())
        for name, checksum in checksums.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != checksum:
                raise ValueError(f"Publisher checksum mismatch for {name}")
        name = next(name for name in checksums if name.endswith("/R39001_data.tsv"))
        lines = archive.read(name).splitlines(keepends=True)
        selected = [lines[0]]
        for line in lines[1:]:
            fields = line.rstrip(b"\r\n").split(b"\t")
            if fields[1] == b"E-10-1966" and fields[11].startswith(b"soil_") and fields[12] in VARIABLES:
                selected.append(line)
        destination.mkdir(exist_ok=True, parents=True)
        (destination / "hogasa_soil.tsv").write_bytes(b"".join(selected))
        docs = {"R39001_README.txt", "R39001plot_metadata.tsv", "R39001units_description.tsv",
                "R39001component_description.tsv", "R39001variables_description.tsv"}
        for name in checksums:
            if Path(name).name in docs:
                (destination / Path(name).name).write_bytes(archive.read(name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Optional original SND ZIP, for offline checks")
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.archive is None:
        with urlopen(URL, timeout=120) as source:
            payload = source.read()
    else:
        payload = args.archive.read_bytes()
    extract(payload, args.output)
