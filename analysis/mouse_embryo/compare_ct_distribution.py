"""R2 Tech8: original vs reconstructed mouse embryo cell-type distribution comparison.

Compares the 30-class cell-type composition between the original E16.5 Stereo-seq
MOSTA data (13 measured slices) and the NTF-reconstructed dense volume, using
proportions (reconstructed data has ~5x more cells due to in-gap densification).

Outputs (written to the experiment directory, cwd):
  - ct_distribution.csv           30-class counts & proportions, both sources
  - ct_distribution_metrics.csv   Pearson/Spearman on proportions + TV distance
  - generation_info.txt           verbatim dump of uns['generation_info']
  - z_freq_long.csv               long table: source x z_bin x cell_type x n x freq
                                  (original: 13 slice layers; reconstructed: ~65 z bins)

Run (env: ntf):
  /home/gongyuqiao/.conda/envs/ntf/bin/python compare_ct_distribution.py
"""

import json

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats

ORIG = "/home/gongyuqiao/ur_annotation/NTF/mytrain/data/mouse_embryo/processed/E16.5_aligned_1030svg.h5ad"
GEN = "/home/gongyuqiao/ur_annotation/NTF/mytrain/data/mouse_embryo/generated/1000w-2/1000w.h5ad"
N_Z_BINS = 65  # z bins for the reconstructed data over the shared z range

# ---------------------------------------------------------------- load
a = ad.read_h5ad(ORIG, backed="r")
orig_ann = a.obs["annotation"].to_numpy()
# original: one z per slice; use median z_scaled as the layer coordinate
orig_z_layer = a.obs.groupby("slice", observed=True)["z_scaled"].median()
orig_slice = a.obs["slice"].to_numpy()
a.file.close()

b = ad.read_h5ad(GEN, backed="r")
gen_ann = b.obs["annotation"].to_numpy()
gen_z = np.asarray(b.obsm["spatial"][:, 2])  # spatial z == z_scaled units (checked)
gen_info = {k: (v.tolist() if isinstance(v, np.ndarray) else
                v.item() if isinstance(v, np.generic) else v)
            for k, v in b.uns["generation_info"].items()}
b.file.close()

cell_types = sorted(set(orig_ann) | set(gen_ann))
assert len(cell_types) == 30, len(cell_types)
print(f"original: {len(orig_ann)} cells; reconstructed: {len(gen_ann)} cells")

# ------------------------------------------------- 1. distribution table
oc = pd.Series(orig_ann).value_counts().reindex(cell_types, fill_value=0)
gc = pd.Series(gen_ann).value_counts().reindex(cell_types, fill_value=0)
dist = pd.DataFrame({
    "cell_type": cell_types,
    "n_original": oc.values,
    "prop_original": oc.values / oc.values.sum(),
    "n_reconstructed": gc.values,
    "prop_reconstructed": gc.values / gc.values.sum(),
})
dist["prop_diff"] = dist["prop_reconstructed"] - dist["prop_original"]
dist["log2_ratio"] = np.log2((dist["prop_reconstructed"] + 1e-6) /
                             (dist["prop_original"] + 1e-6))
dist = dist.sort_values("prop_original", ascending=False).reset_index(drop=True)
dist.to_csv("ct_distribution.csv", index=False)

p, q = dist["prop_original"].values, dist["prop_reconstructed"].values
pear_r, pear_p = stats.pearsonr(p, q)
spear_r, spear_p = stats.spearmanr(p, q)
tv = 0.5 * np.abs(p - q).sum()
metrics = pd.DataFrame({
    "metric": ["pearson_r", "pearson_p", "spearman_r", "spearman_p", "tv_distance",
               "n_original_cells", "n_reconstructed_cells"],
    "value": [pear_r, pear_p, spear_r, spear_p, tv, len(orig_ann), len(gen_ann)],
})
metrics.to_csv("ct_distribution_metrics.csv", index=False)
print(metrics.to_string(index=False))

with open("generation_info.txt", "w") as f:
    f.write(json.dumps(gen_info, indent=2, default=str))
    f.write("\n")

# ------------------------------------------------- 2. per-z cell-type frequency
rows = []

# original: 13 slice layers, z = median z_scaled of the slice
for sl, zc in orig_z_layer.sort_values().items():
    mask = orig_slice == sl
    ann_sl = orig_ann[mask]
    n = mask.sum()
    vc = pd.Series(ann_sl).value_counts().reindex(cell_types, fill_value=0)
    for ct in cell_types:
        rows.append(("original", float(zc), float(zc), float(zc), sl, ct,
                     int(vc[ct]), vc[ct] / n))

# reconstructed: fixed-width z bins over the shared z range
z_lo, z_hi = float(gen_z.min()), float(gen_z.max())
edges = np.linspace(z_lo, z_hi, N_Z_BINS + 1)
bin_idx = np.clip(np.digitize(gen_z, edges) - 1, 0, N_Z_BINS - 1)
for i in range(N_Z_BINS):
    mask = bin_idx == i
    n = mask.sum()
    if n == 0:
        continue
    ann_b = gen_ann[mask]
    vc = pd.Series(ann_b).value_counts().reindex(cell_types, fill_value=0)
    lo, hi = edges[i], edges[i + 1]
    for ct in cell_types:
        rows.append(("reconstructed", float((lo + hi) / 2), float(lo), float(hi),
                     f"bin{i:03d}", ct, int(vc[ct]), vc[ct] / n))

zfreq = pd.DataFrame(rows, columns=["source", "z_center", "z_lo", "z_hi",
                                    "layer", "cell_type", "n_cells", "freq"])
zfreq.to_csv("z_freq_long.csv", index=False)
print(f"z freq table: {zfreq.shape}, original layers={orig_z_layer.size}, "
      f"reconstructed bins={zfreq[zfreq.source=='reconstructed'].z_center.nunique()}")
print("done")
