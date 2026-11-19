"""Fig 2b six-panel qualitative render: noisy / GT / NTF / SpatialZ / FEAST / UniST.

100w scale-test tier, 3 representative genes (first gene of each of the three
distinct global pattern families in the parametric simulator):
  gene_0  = global_x_gradient
  gene_20 = global_y_gradient
  gene_40 = global_center_radial_out
All method files hold the same 479,844 held-out test cells with GT 3D coords in
obsm['spatial'] and predictions in layers['prediction']; noisy/GT come from the
input 100w.h5ad (X = noisy observation, layers['truth'] = ground truth).
Output: one svg per gene (1x6 panels) + png preview, into figures/.
"""

import numpy as np
import anndata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/mnt/experiments/gongyq/analysis_link/ur_annotation/NTF/mytrain/revision/experiment/r2t5_unist_benchmark/figures"
GENES = {"gene_0": "global_x_gradient",
         "gene_20": "global_y_gradient",
         "gene_40": "global_center_radial_out"}
N_SHOW = 40000

base = "/home/gongyuqiao/ur_annotation/NTF/mytrain"
inp = anndata.read_h5ad(f"{base}/data/simulation/scale_test/100w.h5ad")
test = inp[inp.obs["group"] == "test"]
genes = inp.var_names.tolist()

methods = {
    "NTF": anndata.read_h5ad(f"{base}/res/simulation/scale_test/100w/pred.h5ad"),
    "SpatialZ": anndata.read_h5ad("/home/gongyuqiao/ur_annotation/SpatialZ/res/simulation/100w/pred.h5ad"),
    "FEAST": anndata.read_h5ad(f"{base}/res/FEAST/scale_test/100w/test_results.h5ad"),
    "UniST": anndata.read_h5ad(f"{base}/res/UniST/scale_test/100w_inr/test_results.h5ad"),
}

coords = np.asarray(test.obsm["spatial"])
# consistency check: all method files must cover the same test cells in the same order
for name, ad in methods.items():
    assert ad.n_obs == test.n_obs, (name, ad.n_obs, test.n_obs)
    assert (ad.obs["slice_id"].values == test.obs["slice_id"].values).all(), name
    assert np.allclose(np.asarray(ad.obsm["spatial"]), coords), name
print("cell order consistent across all methods")

rng = np.random.default_rng(0)
sub = np.sort(rng.choice(test.n_obs, size=N_SHOW, replace=False))
xyz = coords[sub]

for gene, pat in GENES.items():
    g = genes.index(gene)
    noisy = np.asarray(test.X[:, g]).ravel()
    truth = np.asarray(test.layers["truth"][:, g]).ravel()
    preds = {m: np.asarray(ad.layers["prediction"][:, g]).ravel()
             for m, ad in methods.items()}
    vmax = float(np.percentile(truth, 99))
    panels = [("Noisy observation", noisy), ("Ground truth", truth)] + \
             [(m, preds[m]) for m in methods]

    fig = plt.figure(figsize=(24, 4.2))
    for j, (title, vals) in enumerate(panels):
        ax = fig.add_subplot(1, 6, j + 1, projection="3d")
        v = vals[sub]
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=v, cmap="viridis",
                   s=0.3, vmin=0, vmax=vmax, rasterized=True)
        ax.set_title(title, fontsize=13)
        ax.set_axis_off()
        ax.view_init(elev=20, azim=45)
    fig.suptitle(f"{gene} ({pat}), 1M-cell scale, held-out sections", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2b_{gene}.svg", bbox_inches="tight")
    fig.savefig(f"{OUT}/fig2b_{gene}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{gene} done")
