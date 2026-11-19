"""Marker-based two-level cell annotation for the t-CyCIF CRC1 3D atlas.

Replicates the original paper's approach (Lin et al., Cell 2023, STAR Methods
"Cell-type identification using CyCIF data" + Table S4 cell-type dictionary):

1. Binarize each of the 16 typing markers: 2-component GMM on log1p intensity
   as initialization, then manually adjusted gates informed by the intensity
   histograms (the paper likewise used GMM + manual adjustment; a global 2-
   component GMM provably mis-splits unimodal-with-tail distributions such as
   CD31/FOXP3). One global gate per marker — CRC1 is a single specimen,
   matching the paper's per-sample gating. Ki67 uses the 488 channel (`Ki67`),
   not `Ki67_570` (abandoned in the paper due to poor staining).
2. Hierarchical boolean rule assignment (modified-SYLARAS style):
   level 1: Keratin+ -> Tumor/Epi.; CD31+ -> Endothelial; Desmin+/aSMA+ ->
   Muscle/Fibroblast; CD45+ -> Immune; else Other.
   level 2 (within CD45+): B cells, Treg, PD1+ T helper, DP Lymphocyte,
   PD1+ Tc, Tc cell, Macrophage I-IV / PDL1+ Macrophage, T helper,
   DN Lymphocyte, PDL1+ lymphocyte, Lymphocyte(III).

Outputs (written back into the h5ad, original backed up first):
  - obs['cell_type']      (fine, 21 types per Table S4)
  - obs['cell_type_main'] (coarse, ~9 functional groups; niche-level)
  - obs['<marker>_pos']   (binarization columns for the 16 typing markers)
  - uns['gating_thresholds'] (per-marker log1p gate)

Also writes QC tables to the experiment annotation directory.

Usage:
    conda activate ntf
    python mytrain/scripts/annotate_cycif_crc1.py
"""

import os
import shutil

import anndata
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

H5AD = '/home/gongyuqiao/ur_annotation/NTF/mytrain/data/cycif_crc1_3d/processed/cycif_crc1.h5ad'
BACKUP = '/home/gongyuqiao/ur_annotation/NTF/mytrain/data/cycif_crc1_3d/processed/cycif_crc1_pre_annotation_backup.h5ad'
OUT_DIR = '/home/gongyuqiao/ur_annotation/NTF/mytrain/revision/experiment/cycif_crc1_benchmark/annotation'

TYPING_MARKERS = ['Keratin', 'Ki67', 'CD3', 'CD20', 'CD45RO', 'CD4', 'CD8a',
                  'CD68', 'CD163', 'FOXP3', 'PD1', 'PDL1', 'CD31', 'aSMA',
                  'Desmin', 'CD45']

SEED = 42
GMM_FIT_SAMPLE = 1_000_000

# Manually adjusted gates (log1p scale) from intensity-histogram inspection
# (GMM initialization provably mis-splits these unimodal-with-tail markers;
# the paper likewise manually adjusted GMM initial gates). Markers not listed
# keep their GMM gate (Keratin, CD3 — clear bimodal valleys).
MANUAL_GATES = {
    'Ki67': 7.5,    # right shoulder of broad main peak (proliferating cells)
    'CD20': 7.0,    # right tail (B cells)
    'CD45RO': 7.5,  # right shoulder
    'CD4': 7.5,     # right shoulder
    'CD8a': 7.0,    # right tail
    'CD68': 6.0,    # right shoulder
    'CD163': 6.5,   # right tail
    'FOXP3': 6.5,   # right tail (rare Tregs)
    'PD1': 7.0,     # right tail
    'PDL1': 6.0,    # right shoulder
    'CD31': 6.0,    # tail beyond sharp background peak at ~3.8
    'aSMA': 6.0,    # valley before second mode
    'Desmin': 7.0,  # right shoulder
    'CD45': 7.0,    # valley between background peak and immune hump
}

# fine type -> coarse (niche-level) group
MAIN_MAP = {
    'Tumor/Epi.': 'Tumor/Epi.',
    'Ki67+ Tumor/Epi.': 'Tumor/Epi.',
    'PDL1+ Tumor/Epi.': 'Tumor/Epi.',
    'Endothelial': 'Endothelial',
    'Muscle/Fibroblast': 'Muscle/Fibroblast',
    'B cells': 'B cells',
    'T helper': 'CD4 T',
    'PD1+ T helper': 'CD4 T',
    'Treg': 'CD4 T',
    'DP Lymphocyte': 'CD4 T',
    'Tc cell': 'CD8 T',
    'PD1+ Tc': 'CD8 T',
    'DN Lymphocyte': 'Other lymphocyte',
    'Lymphocyte(III)': 'Other lymphocyte',
    'PDL1+ lymphocyte': 'Other lymphocyte',
    'Macrophage(I)': 'Macrophage',
    'Macrophage(II)': 'Macrophage',
    'Macrophage(III)': 'Macrophage',
    'Macrophage(IV)': 'Macrophage',
    'PDL1+ Macrophage': 'Macrophage',
    'Other': 'Other',
}


def gmm_binarize(x, seed=SEED):
    """2-component GMM on log1p intensity. Returns (positive bool, threshold
    on the log1p scale, two component means)."""
    lx = np.log1p(x)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(lx), size=min(GMM_FIT_SAMPLE, len(lx)), replace=False)
    gmm = GaussianMixture(n_components=2, random_state=seed).fit(lx[idx, None])
    hi = int(np.argmax(gmm.means_.flatten()))
    labels = gmm.predict(lx[:, None])
    pos = labels == hi
    # gate: boundary between the two fitted components (midpoint of the
    # highest negative and lowest positive fitted values is overkill; use the
    # posterior-0.5 crossing approximated by the min log1p of predicted positives)
    thr = float(lx[pos].min()) if pos.any() else np.inf
    return pos, thr, gmm.means_.flatten().tolist()


def assign_cell_types(pos):
    """Hierarchical boolean assignment. `pos` maps marker -> bool array."""
    n = len(next(iter(pos.values())))
    ct = np.full(n, 'Other', dtype=object)

    ker = pos['Keratin']
    immune = (~ker) & (~pos['CD31']) & (~pos['Desmin']) & (~pos['aSMA']) & pos['CD45']

    # --- non-immune compartments (priority order) ---
    ct[ker & pos['PDL1']] = 'PDL1+ Tumor/Epi.'
    ct[ker & ~pos['PDL1'] & pos['Ki67']] = 'Ki67+ Tumor/Epi.'
    ct[ker & ~pos['PDL1'] & ~pos['Ki67']] = 'Tumor/Epi.'
    ct[(~ker) & pos['CD31']] = 'Endothelial'
    ct[(~ker) & (~pos['CD31']) & (pos['Desmin'] | pos['aSMA'])] = 'Muscle/Fibroblast'

    # --- immune compartment (CD45+), priority order ---
    cd4, cd8 = pos['CD4'], pos['CD8a']
    cd68, cd163 = pos['CD68'], pos['CD163']
    assigned = np.zeros(n, dtype=bool)

    def put(mask, name):
        m = immune & ~assigned & mask
        ct[m] = name
        assigned[m] = True

    put(pos['CD20'], 'B cells')
    put(cd4 & pos['FOXP3'], 'Treg')
    put(pos['CD3'] & cd4 & pos['PD1'] & ~cd8, 'PD1+ T helper')
    put(cd4 & cd8, 'DP Lymphocyte')
    put(cd8 & pos['PD1'], 'PD1+ Tc')
    put(cd8 & ~cd4, 'Tc cell')
    # macrophage rules before plain T helper so that CD4+CD68+ -> Macrophage(II)
    put((cd68 | cd163) & pos['PDL1'], 'PDL1+ Macrophage')
    put(cd68 & cd163, 'Macrophage(IV)')
    put(cd4 & cd68 & pos['CD45RO'], 'Macrophage(II)')
    put(cd163 & ~cd68, 'Macrophage(I)')
    put(cd68 & ~cd163, 'Macrophage(III)')
    put(cd4 & ~cd8, 'T helper')
    put(pos['CD3'] & pos['CD45RO'] & ~cd4 & ~cd8, 'DN Lymphocyte')
    put(pos['PDL1'], 'PDL1+ lymphocyte')
    ct[immune & ~assigned] = 'Lymphocyte(III)'  # CD45+ catch-all

    return ct


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'Reading {H5AD} ...')
    adata = anndata.read_h5ad(H5AD)
    print(f'{adata.n_obs:,} cells x {adata.n_vars} markers')

    # --- 1. binarization ---
    pos, thr_rows = {}, []
    for m in TYPING_MARKERS:
        x = np.asarray(adata[:, m].X).flatten().astype(np.float64)
        p_gmm, thr_gmm, means = gmm_binarize(x)
        if m in MANUAL_GATES:
            thr, source = MANUAL_GATES[m], 'manual_adjusted'
            p = np.log1p(x) >= thr
        else:
            thr, source, p = thr_gmm, 'gmm', p_gmm
        pos[m] = p
        thr_rows.append({'marker': m, 'gate_log1p': thr, 'gate_source': source,
                         'gmm_gate_log1p': thr_gmm,
                         'gmm_means_log1p': ';'.join(f'{v:.3f}' for v in sorted(means)),
                         'positive_fraction': float(p.mean())})
        print(f'  {m:10s} gate(log1p)={thr:7.3f} [{source}]  pos_frac={p.mean():.4f}')
    thr_df = pd.DataFrame(thr_rows)
    thr_df.to_csv(os.path.join(OUT_DIR, 'gating_thresholds.csv'), index=False)

    # --- 2. hierarchical assignment ---
    ct = assign_cell_types(pos)
    ct = pd.Categorical(ct, categories=[c for c in MAIN_MAP if c in set(ct)])
    ct_main = pd.Categorical(pd.Series(ct).map(MAIN_MAP).to_numpy())

    # --- 3. write back (backup first) ---
    if not os.path.exists(BACKUP):
        print('Backing up original h5ad ...')
        shutil.copy2(H5AD, BACKUP)
    for m in TYPING_MARKERS:
        adata.obs[f'{m}_pos'] = pos[m]
    adata.obs['cell_type'] = ct
    adata.obs['cell_type_main'] = ct_main
    adata.uns['gating_thresholds'] = {r['marker']: r['gate_log1p'] for _, r in thr_df.iterrows()}
    adata.uns['annotation_note'] = ('GMM binarization (log1p, 2-component, seed=42, global per marker) '
                                    '+ hierarchical boolean gating per Lin et al. Cell 2023 Table S4')
    print('Writing annotated h5ad ...')
    adata.write_h5ad(H5AD, compression='gzip')

    # --- 4. QC tables ---
    counts = (adata.obs.groupby(['cell_type_main', 'cell_type'], observed=True)
              .size().reset_index(name='n_cells'))
    counts['fraction'] = counts['n_cells'] / adata.n_obs
    counts.to_csv(os.path.join(OUT_DIR, 'celltype_counts.csv'), index=False)
    map_df = pd.DataFrame({'cell_type': list(MAIN_MAP), 'cell_type_main': [MAIN_MAP[k] for k in MAIN_MAP]})
    map_df.to_csv(os.path.join(OUT_DIR, 'celltype_main_map.csv'), index=False)
    # per-slice composition (for spatial QC plots)
    per_slice = (adata.obs.groupby(['slice', 'cell_type_main'], observed=True)
                 .size().reset_index(name='n_cells'))
    per_slice.to_csv(os.path.join(OUT_DIR, 'celltype_main_per_slice.csv'), index=False)
    print(counts.to_string(index=False))
    print('Done.')


if __name__ == '__main__':
    main()
