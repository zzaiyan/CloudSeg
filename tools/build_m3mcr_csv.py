import argparse
import csv
import re
from pathlib import Path


PATTERNS = {
    "cloudfree": ("planet_cloudfree_*/*/*.tif", re.compile(r"^planet_cloudfree_(.+_\d+_p_\d+)\.tif$")),
    "cloudy": ("planet_cloudy_*/*/*.tif", re.compile(r"^planet_cloudy_(.+_\d+_p_\d+)\.tif$")),
    "s1": ("S1_*/*/*.tif", re.compile(r"^S1_(.+_\d+_p_\d+)\.tif$")),
    "landcover": ("landcover_*/*/*.tif", re.compile(r"^landcover_(.+_\d+_p_\d+)\.tif$")),
    "cloudmask": ("cloudmask_*/*/*.npy", re.compile(r"^cloudmask_(.+_\d+_p_\d+)\.npy$")),
}


def collect_files(split_root: Path):
    by_modality = {}
    for modality, (glob_pattern, filename_pattern) in PATTERNS.items():
        mapping = {}
        for path in split_root.glob(glob_pattern):
            match = filename_pattern.match(path.name)
            if not match:
                raise ValueError(f"Unexpected filename for {modality}: {path.name}")
            mapping[match.group(1)] = path.relative_to(split_root).as_posix()
        by_modality[modality] = mapping
    return by_modality


def validate_keys(by_modality):
    base_keys = set(by_modality["cloudfree"])
    for modality, mapping in by_modality.items():
        missing = sorted(base_keys - set(mapping))
        extra = sorted(set(mapping) - base_keys)
        if missing or extra:
            raise ValueError(
                f"Key mismatch for {modality}: missing={missing[:3]}, extra={extra[:3]}"
            )
    return sorted(base_keys)


def build_rows(by_modality, keys):
    rows = []
    for key in keys:
        rows.append(
            [
                by_modality["cloudfree"][key],
                by_modality["cloudy"][key],
                by_modality["s1"][key],
                by_modality["landcover"][key],
                by_modality["cloudmask"][key],
            ]
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Build an M3M-CR split CSV from the folder structure.")
    parser.add_argument("--split-root", required=True, help="Path to a split directory, e.g. /data/.../M3M-CR/test")
    parser.add_argument("--output", required=True, help="CSV file to write")
    args = parser.parse_args()

    split_root = Path(args.split_root).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    by_modality = collect_files(split_root)
    keys = validate_keys(by_modality)
    rows = build_rows(by_modality, keys)

    with output_path.open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)

    print(f"split_root={split_root}")
    print(f"rows={len(rows)}")
    print(f"output={output_path}")
    if rows:
        print(f"first_row={rows[0]}")


if __name__ == "__main__":
    main()
