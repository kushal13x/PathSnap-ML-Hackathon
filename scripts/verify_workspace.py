"""Independent workspace verification script for PathSnap Phase 0.

Verifies directory structure, preserved source ZIPs, official reference PDFs,
and raw dataset integrity for both training (80 trips) and student_test (10 trips).
"""

import json
import sys
import zipfile
from pathlib import Path


def verify_workspace(project_root: Path) -> dict:
    """Run all workspace integrity checks and return a structured report."""
    expected_top_dirs = [
        "data",
        "docs",
        "backend",
        "frontend",
        "artifacts",
        "results",
        "visualizations",
        "reports",
        "scripts",
        "tests",
        "configs",
        "source_zips",
    ]
    expected_zips = [
        "Student folder.zip",
        "PathSnap_student_materials.zip",
    ]
    expected_pdfs = [
        "Main Guidelines.pdf",
        "Models and Training Guidelines.pdf",
        "Deliverables.pdf",
        "Assessment and Plagiarism Policy.pdf",
        "Dataset Exploration.pdf",
    ]
    expected_exts = {".track", ".nodes", ".arcs", ".route"}

    report = {
        "project_root": str(project_root),
        "top_level_dirs": {"expected": len(expected_top_dirs), "found": [], "missing": []},
        "source_zips": {"expected": len(expected_zips), "found": {}, "errors": []},
        "reference_pdfs": {"expected": len(expected_pdfs), "found": {}, "errors": []},
        "dataset": {
            "train": {"expected_trips": 80, "found_trips": 0, "valid_trips": 0, "errors": []},
            "student_test": {"expected_trips": 10, "found_trips": 0, "valid_trips": 0, "errors": []},
        },
        "unexpected_root_items": [],
        "integrity_passed": False,
    }

    # 1. Top-level directories
    for d in expected_top_dirs:
        p = project_root / d
        if p.is_dir():
            report["top_level_dirs"]["found"].append(d)
        else:
            report["top_level_dirs"]["missing"].append(d)

    for item in sorted(project_root.iterdir()):
        if item.name.startswith("."):
            continue
        if item.name not in expected_top_dirs:
            report["unexpected_root_items"].append(item.name)

    # 2. Source ZIPs
    zips_dir = project_root / "source_zips"
    for zname in expected_zips:
        zp = zips_dir / zname
        if not zp.is_file():
            report["source_zips"]["errors"].append(f"Missing zip: {zp}")
            continue
        size = zp.stat().st_size
        if size == 0:
            report["source_zips"]["errors"].append(f"Empty zip: {zp}")
            continue
        try:
            with zipfile.ZipFile(zp, "r") as zf:
                bad_member = zf.testzip()
                if bad_member is not None:
                    report["source_zips"]["errors"].append(
                        f"Corrupt member {bad_member} in {zname}"
                    )
                else:
                    report["source_zips"]["found"][zname] = {
                        "size_bytes": size,
                        "entry_count": len(zf.infolist()),
                    }
        except zipfile.BadZipFile as exc:
            report["source_zips"]["errors"].append(f"BadZipFile {zname}: {exc}")

    # 3. Official Reference PDFs
    ref_dir = project_root / "docs" / "reference"
    if not ref_dir.is_dir():
        report["reference_pdfs"]["errors"].append("Missing docs/reference directory")
    else:
        for pdf_name in expected_pdfs:
            pp = ref_dir / pdf_name
            if not pp.is_file():
                report["reference_pdfs"]["errors"].append(f"Missing PDF: {pdf_name}")
            elif pp.stat().st_size == 0:
                report["reference_pdfs"]["errors"].append(f"Empty PDF: {pdf_name}")
            else:
                # Verify PDF header signature
                with open(pp, "rb") as f:
                    header = f.read(5)
                if header != b"%PDF-":
                    report["reference_pdfs"]["errors"].append(
                        f"Invalid PDF signature in {pdf_name}: {header!r}"
                    )
                else:
                    report["reference_pdfs"]["found"][pdf_name] = pp.stat().st_size

        extra_ref = [
            f.name for f in ref_dir.iterdir() if f.name not in expected_pdfs
        ]
        if extra_ref:
            report["reference_pdfs"]["errors"].append(
                f"Unexpected files/dirs in docs/reference: {extra_ref}"
            )

    # 4. Dataset Splits
    for split, expected_count in [("train", 80), ("student_test", 10)]:
        split_dir = project_root / "data" / "raw" / split
        split_rep = report["dataset"][split]
        if not split_dir.is_dir():
            split_rep["errors"].append(f"Missing split directory: {split_dir}")
            continue

        loose_files = [f.name for f in split_dir.iterdir() if not f.is_dir()]
        if loose_files:
            split_rep["errors"].append(f"Loose non-directory items: {loose_files}")

        trip_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        split_rep["found_trips"] = len(trip_dirs)
        if len(trip_dirs) != expected_count:
            split_rep["errors"].append(
                f"Expected {expected_count} trips, found {len(trip_dirs)}"
            )

        for td in trip_dirs:
            trip_id = td.name
            items = list(td.iterdir())
            nested_dirs = [x.name for x in items if x.is_dir()]
            if nested_dirs:
                split_rep["errors"].append(
                    f"{split}/{trip_id} has nested dirs: {nested_dirs}"
                )
                continue

            files = [x for x in items if x.is_file()]
            if len(files) != 4:
                split_rep["errors"].append(
                    f"{split}/{trip_id} has {len(files)} files (expected 4)"
                )
                continue

            exts = set()
            trip_ok = True
            for tf in files:
                if tf.stat().st_size == 0:
                    split_rep["errors"].append(f"{split}/{trip_id}/{tf.name} is empty")
                    trip_ok = False
                if tf.stem != trip_id:
                    split_rep["errors"].append(
                        f"{split}/{trip_id}/{tf.name} stem mismatch"
                    )
                    trip_ok = False
                exts.add(tf.suffix)

            if exts != expected_exts:
                split_rep["errors"].append(
                    f"{split}/{trip_id} extensions mismatch: {sorted(exts)}"
                )
                trip_ok = False

            if trip_ok:
                split_rep["valid_trips"] += 1

    all_errors = (
        report["top_level_dirs"]["missing"]
        + report["source_zips"]["errors"]
        + report["reference_pdfs"]["errors"]
        + report["dataset"]["train"]["errors"]
        + report["dataset"]["student_test"]["errors"]
    )
    report["integrity_passed"] = len(all_errors) == 0
    return report


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    report = verify_workspace(project_root)

    print("=== PATHSNAP WORKSPACE VERIFICATION ===")
    print(f"Project Root: {report['project_root']}")
    print(
        f"Top-level directories: {len(report['top_level_dirs']['found'])}/{report['top_level_dirs']['expected']}"
    )
    print(f"Source ZIPs verified (CRC checked): {len(report['source_zips']['found'])}/2")
    for name, info in sorted(report["source_zips"]["found"].items()):
        print(f"  - {name}: {info['size_bytes']} bytes, {info['entry_count']} entries")
    print(f"Reference PDFs verified: {len(report['reference_pdfs']['found'])}/5")
    for name, size in sorted(report["reference_pdfs"]["found"].items()):
        print(f"  - {name}: {size} bytes")

    for split in ["train", "student_test"]:
        s = report["dataset"][split]
        print(
            f"Dataset [{split}]: {s['valid_trips']}/{s['expected_trips']} valid trips "
            f"(found_dirs={s['found_trips']}, errors={len(s['errors'])})"
        )

    if report["unexpected_root_items"]:
        print(f"Unexpected root items: {report['unexpected_root_items']}")

    print(f"INTEGRITY_PASSED: {report['integrity_passed']}")
    if not report["integrity_passed"]:
        print("ERRORS:")
        print(json.dumps(report, indent=2))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
