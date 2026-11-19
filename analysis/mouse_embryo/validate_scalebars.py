import argparse
import re
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd
from PIL import Image


NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def child_by_id(root, element_id):
    for element in root.iter():
        if element.attrib.get("id") == element_id:
            return element
    raise KeyError(element_id)


def local_name(tag):
    return tag.rsplit("}", 1)[-1]


def validate_row(row):
    svg_path = Path(row["svg_file"])
    png_path = Path(row["png_file"])
    root = ET.parse(svg_path).getroot()
    line_group = child_by_id(root, row["scalebar_line_svg_id"])
    path = next(element for element in line_group.iter() if local_name(element.tag) == "path")
    match = re.search(
        rf"M\s*({NUMBER})\s+({NUMBER})\s*L\s*({NUMBER})\s+({NUMBER})",
        path.attrib["d"],
    )
    if match is None:
        raise ValueError(f"Cannot parse scale-bar path: {path.attrib['d']}")
    x1, y1, x2, y2 = map(float, match.groups())
    line_length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    clip_id = re.search(r"url\(#([^)]+)\)", path.attrib["clip-path"]).group(1)
    clip_path = child_by_id(root, clip_id)
    clip_rect = next(element for element in clip_path.iter() if local_name(element.tag) == "rect")
    axis_width = float(clip_rect.attrib["width"])
    actual_fraction = line_length / axis_width
    expected_fraction = float(row["scalebar_fraction_of_xrange"])
    relative_error = abs(actual_fraction - expected_fraction) / expected_fraction
    text_group = child_by_id(root, row["scalebar_text_svg_id"])
    svg_label = "".join(text_group.itertext()).strip()
    with Image.open(png_path) as image:
        image.verify()
    with Image.open(png_path) as image:
        png_width, png_height = image.size
    passed = relative_error < 1e-6 and svg_label == row["scalebar_label"]
    return {
        "dataset": row["dataset"],
        "panel": row["panel"],
        "scalebar_label": row["scalebar_label"],
        "svg_label": svg_label,
        "svg_line_length_pt": line_length,
        "axis_clip_width_pt": axis_width,
        "expected_fraction": expected_fraction,
        "actual_svg_fraction": actual_fraction,
        "relative_error": relative_error,
        "png_width_px": png_width,
        "png_height_px": png_height,
        "status": "PASS" if passed else "FAIL",
    }


def main():
    parser = argparse.ArgumentParser(description="Independently validate scale-bar geometry in generated SVG files")
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("/home/gongyuqiao/ur_annotation/NTF/mytrain/revision/experiment/r2m3_spatial_scalebars"),
    )
    args = parser.parse_args()
    experiment_dir = args.experiment_dir.resolve()
    manifest_path = experiment_dir / "scalebar_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    results = pd.DataFrame([validate_row(row) for _, row in manifest.iterrows()])
    results.to_csv(experiment_dir / "validation_summary.csv", index=False)
    excel_path = experiment_dir / "scalebar_manifest.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        results.to_excel(writer, sheet_name="validation", index=False)
    workbook = pd.read_excel(excel_path, sheet_name=None)
    if set(workbook) != {"scalebars", "datasets", "validation"}:
        raise RuntimeError(f"Unexpected workbook sheets: {list(workbook)}")
    if len(workbook["scalebars"]) != 14 or len(workbook["datasets"]) != 7 or len(workbook["validation"]) != 14:
        raise RuntimeError("Workbook row counts do not match 7 datasets and 14 panels")
    if not (results["status"] == "PASS").all():
        raise RuntimeError(results[results["status"] != "PASS"].to_string(index=False))
    print(results[["dataset", "panel", "scalebar_label", "actual_svg_fraction", "status"]].to_string(index=False))
    print(f"All {len(results)} panels passed SVG geometry, label, PNG, and workbook validation")


if __name__ == "__main__":
    main()
