import argparse
import math
from pathlib import Path
from xml.etree import ElementTree as ET

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from scipy import sparse


DATASETS = [
    {
        "dataset": "mouseVC",
        "title": "STARmap mouse visual cortex",
        "input": "mytrain/data/mouse_VC/processed/mouseVC_28gene.h5ad",
        "slice_key": "slice_id",
        "slice_value": "slice_2",
        "slice_label": "slice_2",
        "coord_key": "spatial",
        "annotation_key": "leiden",
        "annotation_title": "Leiden cluster",
        "gene": "Rorb",
        "xy_factor": 1.0,
        "source_unit": "pixels",
        "scale_unit": "pixels",
        "z_mode": "none",
        "unit_evidence": "Raw STARmap_mouse_VC_3D_data.h5ad stores integer x/y image coordinates without physical pixel-size metadata; retained as pixels.",
    },
    {
        "dataset": "drosophila",
        "title": "Drosophila E16-18h Stereo-seq",
        "input": "mytrain/data/drosophila/processed/E16-18h_735avg.h5ad",
        "slice_key": "slice_ID",
        "slice_value": "E16-18h_a_S04",
        "slice_label": "E16-18h_a_S04",
        "coord_key": "spatial",
        "annotation_key": "annotation",
        "annotation_title": "Tissue annotation",
        "gene": "grh",
        "xy_factor": 10.0,
        "source_unit": "bin20 coordinate units",
        "scale_unit": "µm",
        "z_mode": "physical",
        "unit_evidence": "Wang et al. (2022), doi:10.1016/j.devcel.2022.04.006: E16-18h uses bin20 = 10 µm; new_x/new_y/new_z are bin20-aligned coordinates, so multiply by 10 µm/unit.",
    },
    {
        "dataset": "imc_bc",
        "title": "IMC HER2+ breast cancer",
        "input": "mytrain/data/ICM_BC/processed/imc_bc_annotated.h5ad",
        "slice_key": "z_slice",
        "slice_value": 75,
        "slice_label": "optical layer 75",
        "coord_key": "spatial",
        "annotation_key": "cell_type_main",
        "annotation_title": "Cell type",
        "gene": "panCK",
        "xy_factor": 1.0,
        "source_unit": "pixels",
        "scale_unit": "pixels",
        "z_mode": "none",
        "unit_evidence": "imc_bc_benchmark README and obs x_pixel/y_pixel: acquisition pixel size is unavailable; retained as pixels.",
    },
    {
        "dataset": "mouse_embryo",
        "title": "MOSTA E16.5 mouse embryo Stereo-seq",
        "input": "mytrain/data/mouse_embryo/processed/E16.5_aligned_1030svg.h5ad",
        "slice_key": "slice",
        "slice_value": "slice7",
        "slice_label": "slice7",
        "coord_key": "spatial",
        "annotation_key": "annotation",
        "annotation_title": "Tissue annotation",
        "gene": "Otx2",
        "xy_factor": 25.0,
        "source_unit": "bin50 coordinate units",
        "scale_unit": "µm",
        "z_mode": "physical",
        "unit_evidence": "Chen et al. (2022), doi:10.1016/j.cell.2022.04.003: MOSTA uses bin50 (50 × 0.5 µm DNB pitch = 25 µm/unit); mouseEmbryo.ipynb sets z_scaled=z/25, confirming the same coordinate unit.",
    },
    {
        "dataset": "deep_starmap",
        "title": "Deep-STARmap mouse brain",
        "input": "mytrain/data/deep_starmap_brain/processed/deep_starmap_brain.h5ad",
        "slice_key": "slice_id",
        "slice_value": 7,
        "slice_label": "merged slice 7 (optical layers 97-112)",
        "coord_key": "spatial",
        "annotation_key": "FUSEmap_main_level",
        "annotation_title": "FUSEmap main level",
        "gene": "PCP4",
        "xy_factor": 0.32,
        "source_unit": "pixels",
        "scale_unit": "µm",
        "z_mode": "none",
        "unit_evidence": "deep_starmap_brain README: voxel size 320 × 320 × 700 nm³; xy pixels multiplied by 0.32 µm/pixel.",
    },
    {
        "dataset": "wemerfish",
        "title": "weMERFISH zebrafish embryo C_6s_E2",
        "input": "mytrain/data/wemerfish_zebrafish_embryo/processed/wemerfish_C6s_E2.h5ad",
        "slice_key": "slice_id",
        "slice_value": 8,
        "slice_label": "virtual slice 8",
        "coord_key": "spatial",
        "annotation_key": "type",
        "annotation_title": "Niche annotation",
        "gene": "sox19a",
        "xy_factor": 1.0,
        "source_unit": "µm",
        "scale_unit": "µm",
        "z_mode": "physical",
        "unit_evidence": "Dryad README_dryad.md: native whole-mount spatial coordinates with z rescaled to real-world sample thickness; existing benchmark records z in µm on the same Cartesian scale as x/y.",
    },
    {
        "dataset": "t_cycif",
        "title": "t-CyCIF CRC1 colorectal cancer",
        "input": "mytrain/data/cycif_crc1_3d/processed/cycif_crc1.h5ad",
        "slice_key": "slice_id",
        "slice_value": 50,
        "slice_label": "section 50",
        "coord_key": "spatial_aligned_um",
        "annotation_key": "cell_type_main",
        "annotation_title": "Cell type",
        "gene": "Keratin",
        "xy_factor": 1.0,
        "source_unit": "µm",
        "scale_unit": "µm",
        "z_mode": "physical",
        "unit_evidence": "cycif_crc1_3d README and uns pixel_size_um: registered xy pixels converted using 0.65 µm/pixel; spatial_aligned_um is already in µm.",
    },
]


def nice_length(target):
    if target <= 0:
        raise ValueError("Scale-bar target must be positive")
    exponent = math.floor(math.log10(target))
    base = 10 ** exponent
    fraction = target / base
    nice = min([1, 2, 5, 10], key=lambda value: abs(value - fraction))
    return nice * base


def format_number(value):
    if np.isclose(value, round(value)):
        return str(int(round(value)))
    return f"{value:.3g}"


def nonzero_minmax(values):
    values = np.asarray(values, dtype=float)
    result = np.zeros_like(values)
    mask = values > 1e-6
    if not mask.any():
        return result, None, None
    minimum = float(values[mask].min())
    maximum = float(values[mask].max())
    if maximum > minimum:
        result[mask] = 0.1 + 0.9 * (values[mask] - minimum) / (maximum - minimum)
    else:
        result[mask] = 1.0
    return np.clip(result, 0, 1), minimum, maximum


def load_slice(root, config):
    path = root / config["input"]
    adata = ad.read_h5ad(path, backed="r")
    try:
        if config["slice_key"] not in adata.obs:
            raise KeyError(f"Missing slice column: {config['slice_key']}")
        if config["annotation_key"] not in adata.obs:
            raise KeyError(f"Missing annotation column: {config['annotation_key']}")
        if config["coord_key"] not in adata.obsm:
            raise KeyError(f"Missing coordinate key: {config['coord_key']}")
        if config["gene"] not in adata.var_names:
            raise KeyError(f"Missing gene/marker: {config['gene']}")
        mask = np.asarray(adata.obs[config["slice_key"]] == config["slice_value"])
        if not mask.any():
            raise ValueError(f"Empty slice: {config['slice_value']}")
        coords = np.asarray(adata.obsm[config["coord_key"]])[mask, :3].astype(float)
        labels = adata.obs[config["annotation_key"]].to_numpy()[mask]
        gene_index = adata.var_names.get_loc(config["gene"])
        expression = adata.X[:, gene_index]
        if sparse.issparse(expression):
            expression = expression.toarray()
        expression = np.asarray(expression).ravel()[mask]
    finally:
        adata.file.close()
    if not np.isfinite(coords).all() or not np.isfinite(expression).all():
        raise ValueError(f"Non-finite coordinate or expression in {config['dataset']}")
    coords[:, :2] *= config["xy_factor"]
    labels = pd.Series(labels)
    labels = labels.where(labels.notna(), "Unassigned").astype(str).to_numpy()
    normalized, minimum, maximum = nonzero_minmax(expression)
    z = coords[:, 2].copy()
    if config["z_mode"] == "physical":
        z *= config["xy_factor"]
    return path, coords, z, labels, expression, normalized, minimum, maximum


def point_size(n):
    if n <= 3000:
        return 12
    if n <= 10000:
        return 6
    if n <= 30000:
        return 3
    return 1.2


def category_palette(categories):
    colors = []
    for cmap_name in ["tab20", "tab20b", "tab20c"]:
        cmap = plt.get_cmap(cmap_name)
        colors.extend([cmap(i) for i in range(cmap.N)])
    if len(categories) > len(colors):
        colors = [plt.get_cmap("hsv")(i / len(categories)) for i in range(len(categories))]
    return dict(zip(categories, colors))


def add_scalebar(ax, xlim, ylim, unit, gid):
    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]
    length = nice_length(0.2 * x_span)
    x_end = xlim[1] - 0.045 * x_span
    x_start = x_end - length
    y_bar = ylim[0] + 0.065 * y_span
    pad_x = 0.012 * x_span
    box = FancyBboxPatch(
        (x_start - pad_x, ylim[0] + 0.018 * y_span),
        length + 2 * pad_x,
        0.105 * y_span,
        boxstyle="round,pad=0.01",
        facecolor="white",
        edgecolor="none",
        alpha=0.88,
        zorder=5,
    )
    ax.add_patch(box)
    line = ax.plot(
        [x_start, x_end],
        [y_bar, y_bar],
        color="black",
        linewidth=2.2,
        solid_capstyle="butt",
        zorder=7,
    )[0]
    line.set_gid(f"{gid}_line")
    tick_height = 0.012 * y_span
    for x in [x_start, x_end]:
        ax.plot([x, x], [y_bar - tick_height, y_bar + tick_height], color="black", linewidth=1.2, zorder=7)
    label = f"{format_number(length)} {unit}"
    text = ax.text(
        (x_start + x_end) / 2,
        y_bar + 0.024 * y_span,
        label,
        ha="center",
        va="bottom",
        fontsize=9,
        color="black",
        zorder=7,
    )
    text.set_gid(f"{gid}_text")
    return length, label


def z_detail(config, z):
    if config["z_mode"] == "none":
        return ""
    if np.isclose(z.min(), z.max()):
        return f"; z={format_number(z.min())} µm"
    return f"; z={format_number(z.min())} to {format_number(z.max())} µm"


def plot_dataset(root, output_dir, config, seed=42, max_points=60000):
    path, coords, z, labels, expression, normalized, minimum, maximum = load_slice(root, config)
    n_total = len(coords)
    rng = np.random.default_rng(seed)
    selected = rng.choice(n_total, size=min(n_total, max_points), replace=False)
    selected.sort()
    x, y = coords[selected, 0], coords[selected, 1]
    x_span, y_span = np.ptp(coords[:, 0]), np.ptp(coords[:, 1])
    if x_span <= 0 or y_span <= 0:
        raise ValueError(f"Degenerate xy extent in {config['dataset']}")
    right_padding = 0.35 if y_span / x_span > 1.4 else 0.025
    xlim = (coords[:, 0].min() - 0.025 * x_span, coords[:, 0].max() + right_padding * x_span)
    ylim = (coords[:, 1].min() - 0.025 * y_span, coords[:, 1].max() + 0.025 * y_span)
    categories = sorted(np.unique(labels).tolist())
    palette = category_palette(categories)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.8))
    panel_records = []
    for panel, ax in zip(["annotation", "expression"], axes):
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor("white")
        if panel == "annotation":
            for category in categories:
                mask = labels[selected] == category
                ax.scatter(x[mask], y[mask], s=point_size(len(selected)), c=[palette[category]], linewidths=0, rasterized=False)
            ax.set_title(f"{config['annotation_title']}\n{len(selected):,}/{n_total:,} spots shown", fontsize=11)
        else:
            order = np.argsort(normalized[selected], kind="stable")
            scatter = ax.scatter(
                x[order],
                y[order],
                c=normalized[selected][order],
                s=point_size(len(selected)),
                cmap="Blues",
                vmin=0,
                vmax=1,
                linewidths=0,
                rasterized=False,
            )
            ax.set_title(f"{config['gene']} expression\n{len(selected):,}/{n_total:,} spots shown", fontsize=11)
            colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 0.5, 1])
            colorbar.set_label("Display-normalized expression", fontsize=9)
            colorbar.ax.tick_params(labelsize=8)
        gid = f"{config['dataset']}_{panel}_scalebar"
        length, label = add_scalebar(ax, xlim, ylim, config["scale_unit"], gid)
        panel_records.append(
            {
                "dataset": config["dataset"],
                "panel": panel,
                "input_file": str(path),
                "slice_key": config["slice_key"],
                "slice_value": str(config["slice_value"]),
                "slice_label": config["slice_label"] + z_detail(config, z),
                "n_spots_total": n_total,
                "n_spots_displayed": len(selected),
                "annotation_key": config["annotation_key"] if panel == "annotation" else "",
                "gene_or_marker": config["gene"] if panel == "expression" else "",
                "expression_source_min": minimum if panel == "expression" else np.nan,
                "expression_source_max": maximum if panel == "expression" else np.nan,
                "coord_key": config["coord_key"],
                "source_unit": config["source_unit"],
                "xy_multiplier_to_display_unit": config["xy_factor"],
                "scale_unit": config["scale_unit"],
                "x_min": xlim[0],
                "x_max": xlim[1],
                "y_min": ylim[0],
                "y_max": ylim[1],
                "scalebar_length": length,
                "scalebar_label": label,
                "scalebar_fraction_of_xrange": length / (xlim[1] - xlim[0]),
                "scalebar_line_svg_id": f"{gid}_line",
                "scalebar_text_svg_id": f"{gid}_text",
                "unit_evidence": config["unit_evidence"],
            }
        )
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=palette[category], markersize=5, label=category)
        for category in categories
    ]
    ncol = min(5, max(3, math.ceil(len(categories) / 4)))
    nrows = math.ceil(len(categories) / ncol)
    bottom = min(0.34, 0.11 + 0.026 * nrows)
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=ncol,
        frameon=False,
        fontsize=8,
        title=config["annotation_title"],
        title_fontsize=9,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    fig.suptitle(f"{config['title']} | measured 2D section: {config['slice_label']}{z_detail(config, z)}", fontsize=14, y=0.97)
    fig.subplots_adjust(left=0.04, right=0.96, top=0.84, bottom=bottom, wspace=0.24)
    svg_path = output_dir / "figures" / f"{config['dataset']}_spatial_scalebar.svg"
    png_path = output_dir / "figures" / f"{config['dataset']}_spatial_scalebar.png"
    fig.savefig(svg_path, format="svg", facecolor="white")
    fig.savefig(png_path, format="png", dpi=300, facecolor="white")
    plt.close(fig)
    root_xml = ET.parse(svg_path).getroot()
    svg_ids = {element.attrib.get("id") for element in root_xml.iter()}
    for record in panel_records:
        record["svg_file"] = str(svg_path)
        record["png_file"] = str(png_path)
        if record["scalebar_line_svg_id"] not in svg_ids or record["scalebar_text_svg_id"] not in svg_ids:
            raise RuntimeError(f"Scale-bar SVG elements missing: {record['scalebar_line_svg_id']}")
    print(f"{config['dataset']}: {n_total:,} spots; scale bar {panel_records[0]['scalebar_label']}; {svg_path.name}")
    return panel_records


def main():
    parser = argparse.ArgumentParser(description="Draw measured 2D spatial plots with verified scale bars")
    parser.add_argument("--root", type=Path, default=Path("/home/gongyuqiao/ur_annotation/NTF"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-points", type=int, default=60000)
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir or root / "mytrain/revision/experiment/r2m3_spatial_scalebars"
    output_dir = output_dir.resolve()
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "svg.fonttype": "none",
            "svg.hashsalt": "r2m3-spatial-scalebars",
        }
    )
    records = []
    for config in DATASETS:
        records.extend(plot_dataset(root, output_dir, config, seed=args.seed, max_points=args.max_points))
    manifest = pd.DataFrame(records)
    manifest.to_csv(output_dir / "scalebar_manifest.csv", index=False)
    with pd.ExcelWriter(output_dir / "scalebar_manifest.xlsx", engine="openpyxl") as writer:
        manifest.to_excel(writer, sheet_name="scalebars", index=False)
        pd.DataFrame(
            [
                {
                    "dataset": config["dataset"],
                    "input": config["input"],
                    "slice": config["slice_label"],
                    "gene_or_marker": config["gene"],
                    "unit_evidence": config["unit_evidence"],
                }
                for config in DATASETS
            ]
        ).to_excel(writer, sheet_name="datasets", index=False)
    print(f"Wrote {len(manifest)} panel records to {output_dir}")


if __name__ == "__main__":
    main()
