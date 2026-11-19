"""Expand the archived paid selections and score arrays alongside each site configuration."""
from pathlib import Path
import re
from zipfile import ZipFile


def main():
    root = Path(__file__).resolve().parent
    collections = sorted(root.rglob("selections.zip"))
    count = 0
    for collection in collections:
        with ZipFile(collection) as archive:
            for member in archive.namelist():
                if not re.fullmatch(r"draw_\d{3,4}\.(npz|json)", member):
                    raise ValueError(f"Unexpected selection record {member!r}")
                path = collection.parent / member
                data = archive.read(member)
                if path.exists():
                    if path.read_bytes() != data:
                        raise FileExistsError(f"Existing selection differs from the archive: {path}")
                else:
                    temp = path.with_suffix(path.suffix + ".part")
                    temp.write_bytes(data)
                    temp.replace(path)
                count += 1
    print(f"{count} selection and pilot-parameter files available in {len(collections)} collections.")


if __name__ == "__main__":
    main()
