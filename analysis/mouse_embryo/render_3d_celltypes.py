"""R2 Tech8: 3D render of representative cell types, original vs reconstructed.

Per representative cell type: 2 rows (original / NTF-reconstructed) x 3 views
(front / side / top). Each panel shows a transparent 3D embryo surface (marching
cubes on a voxelized occupancy grid of the FULL point set of that dataset) plus
the target cell type as a vivid scatter. No background point cloud, so the organ
is never occluded.

Sampling (density-honest): the reconstructed file holds 10M cells but represents
the true 100M-scale volume, so we sample 10% of the 10M file (= 1% of the 100M
scale, ~709k points) against 1% of the original 1.4M data (~14k points); both
thus show the same 1% of their true scale and the ~5x in-gap densification of
the 10M file translates into the same point-density ratio at equal true-scale
sampling. Same seed (42); per-type and overview figures share the samples.

The overview figure keeps the all-30-type colored scatter (colors are the
content); to avoid overplotting it uses 0.2% of each true scale (original 0.2%
of 1.4M, reconstructed 2% of the 10M file = 0.2% of 100M), keeping the 1:10
point-density correspondence.

Outputs: figures/3d_<type>.png and figures/3d_overview.png (dpi=250).

Run (env: ntf, CPU):
  /home/gongyuqiao/.conda/envs/ntf/bin/python render_3d_celltypes.py
"""

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes

ORIG = "/home/gongyuqiao/ur_annotation/NTF/mytrain/data/mouse_embryo/processed/E16.5_aligned_1030svg.h5ad"
GEN = "/home/gongyuqiao/ur_annotation/NTF/mytrain/data/mouse_embryo/generated/1000w-2/1000w.h5ad"
OUT = "figures"

REP_TYPES = ["Brain", "Spinal cord", "Eye", "Inner ear", "Heart",
             "Liver", "Lung", "Kidney", "GI tract", "Meninges"]
FRAC_ORIG = 0.01        # 1% of the original 1.4M data
FRAC_GEN = 0.10         # 10% of the 10M file = 1% of the true 100M scale
SURF_COLOR = "lightgrey"
SURF_ALPHA = 0.15
N_VOX = 72              # voxels along the longest axis for the surface grid
DPI = 250
SEED = 42
rng = np.random.default_rng(SEED)

# fixed shared palette for all 30 types
a = ad.read_h5ad(ORIG, backed="r")
CELL_TYPES = sorted(a.obs["annotation"].unique().tolist())
a.file.close()
cmap = (plt.get_cmap("tab20").colors + plt.get_cmap("tab20b").colors +
        plt.get_cmap("tab20c").colors)
PALETTE = {ct: cmap[i % len(cmap)] for i, ct in enumerate(CELL_TYPES)}

# high-contrast colors for the per-type figures (the light tab20 shades wash out
# on the grey embryo context); same color for a type in both datasets
REP_COLORS = {"Brain": "#2ca02c", "Spinal cord": "#9467bd", "Eye": "#e377c2",
              "Inner ear": "#1f77b4", "Heart": "#d62728", "Liver": "#17becf",
              "Lung": "#ff7f0e", "Kidney": "#8c564b", "GI tract": "#9e9a1d",
              "Meninges": "#000080"}

VIEWS = [("front", dict(elev=10, azim=-60)),
         ("side", dict(elev=10, azim=30)),
         ("top", dict(elev=90, azim=-60))]


def type_point_size(n_sel):
    # small structures get larger points so they stay visible
    return float(np.clip(3000 / max(n_sel, 1), 0.4, 12.0))


def load_source(path):
    adata = ad.read_h5ad(path, backed="r")
    ann = adata.obs["annotation"].to_numpy()
    xyz = np.asarray(adata.obsm["spatial"])
    adata.file.close()
    return ann, xyz


def set_embryo_box(ax):
    ax.set_box_aspect((354, 571, 156))  # x, y, z spans of the aligned volume
    ax.set_axis_off()


print("loading original ...")
ann_o, xyz_o = load_source(ORIG)
print("loading reconstructed ...")
ann_g, xyz_g = load_source(GEN)

# ------------------------------------------------- embryo surface (full data)
# shared voxel grid over the union bounding box; surface per dataset from its
# FULL point set (context only, not subject to the scatter sampling)
lo = np.minimum(xyz_o.min(0), xyz_g.min(0))
hi = np.maximum(xyz_o.max(0), xyz_g.max(0))
step = (hi - lo).max() / N_VOX
shape = tuple(np.ceil((hi - lo) / step).astype(int) + 3)


def embryo_surface(xyz, sigma, level):
    grid = np.zeros(shape, dtype=np.float32)
    ij = np.floor((xyz - lo) / step).astype(int) + 1
    np.add.at(grid, tuple(ij.T), 1.0)
    vol = gaussian_filter(grid, sigma=sigma)
    verts, faces, _, _ = marching_cubes(vol, level=level)
    return verts * step + lo - step, faces


print("building surfaces from full point sets ...")
SURF = {"original": embryo_surface(xyz_o, sigma=1.2, level=0.4),
        "reconstructed": embryo_surface(xyz_g, sigma=1.0, level=0.3)}
for src, (v, f) in SURF.items():
    print(f"  {src} surface: {len(v)} verts, {len(f)} faces")

# ------------------------------------------------- density-honest samples
sel_o = np.sort(rng.choice(xyz_o.shape[0], size=round(FRAC_ORIG * xyz_o.shape[0]),
                           replace=False))
sel_g = np.sort(rng.choice(xyz_g.shape[0], size=round(FRAC_GEN * xyz_g.shape[0]),
                           replace=False))
ann_o_s, xyz_o_s = ann_o[sel_o], xyz_o[sel_o]
ann_g_s, xyz_g_s = ann_g[sel_g], xyz_g[sel_g]
print(f"samples: original {sel_o.size:,} (1% of 1.4M); "
      f"reconstructed {sel_g.size:,} (10% of 10M file = 1% of 100M)")

SRC_LABEL = {"original": "1% of measured 1.4M",
             "reconstructed": "10% of 10M preview; full reconstruction = 100M"}

# ---------------------------------------------------------- per-type figures
for ct in REP_TYPES:
    fig = plt.figure(figsize=(13, 8.6))
    for r, (src, ann, xyz) in enumerate([("original", ann_o_s, xyz_o_s),
                                         ("reconstructed", ann_g_s, xyz_g_s)]):
        pts = xyz[ann == ct]
        s_ct = type_point_size(pts.shape[0])
        v, f = SURF[src]
        for c, (vname, view) in enumerate(VIEWS):
            ax = fig.add_subplot(2, 3, r * 3 + c + 1, projection="3d")
            ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=f,
                            color=SURF_COLOR, alpha=SURF_ALPHA, shade=False)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=s_ct,
                       color=REP_COLORS[ct], alpha=0.7, depthshade=False,
                       rasterized=True)
            ax.view_init(**view)
            set_embryo_box(ax)
            ax.set_title(f"{src} - {vname} (n={pts.shape[0]:,})\n{SRC_LABEL[src]}",
                         fontsize=11)
    fig.suptitle(f"{ct}: original vs reconstructed", fontsize=15)
    fig.tight_layout()
    fn = f"{OUT}/3d_{ct.replace(' ', '_')}.png"
    fig.savefig(fn, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {fn}")

# ---------------------------------------------------------- overview figure
# all-30-type colored scatter; 709k points overplot into a muddy mass, so the
# overview uses 0.2%/2% (still 1:10, i.e. 0.2% of each true scale)
ov_o = np.sort(rng.choice(xyz_o.shape[0], size=round(0.002 * xyz_o.shape[0]),
                          replace=False))
ov_g = np.sort(rng.choice(xyz_g.shape[0], size=round(0.02 * xyz_g.shape[0]),
                          replace=False))
OV_LABEL = {"original": "0.2% of measured 1.4M",
            "reconstructed": "2% of 10M preview; full reconstruction = 100M"}
order = [ct for ct, _ in sorted(
    ((ct, int((ann_o == ct).sum())) for ct in CELL_TYPES), key=lambda t: -t[1])]
fig = plt.figure(figsize=(14, 7))
for j, (src, ann, xyz) in enumerate([("original", ann_o[ov_o], xyz_o[ov_o]),
                                     ("reconstructed", ann_g[ov_g], xyz_g[ov_g])]):
    ax = fig.add_subplot(1, 2, j + 1, projection="3d")
    for ct in order:
        m = ann == ct
        if not m.any():
            continue
        pts = xyz[m]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.6, color=PALETTE[ct],
                   alpha=0.6, label=ct, depthshade=False, rasterized=True)
    ax.view_init(elev=10, azim=30)
    set_embryo_box(ax)
    ax.set_title(f"{src} (n={xyz.shape[0]:,})\n{OV_LABEL[src]}", fontsize=12)
handles = [plt.Line2D([], [], marker="o", ls="", color=PALETTE[ct], markersize=5)
           for ct in CELL_TYPES]
fig.legend(handles, CELL_TYPES, loc="center right", fontsize=7, ncol=2,
           frameon=False, handletextpad=0.1, borderaxespad=0.5)
fig.suptitle("Mouse embryo E16.5: all 30 cell types, original vs reconstructed", fontsize=15)
fig.tight_layout(rect=(0, 0, 0.82, 1))
fn = f"{OUT}/3d_overview.png"
fig.savefig(fn, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"saved {fn}")
print("all 3D renders done")
