"""Preprocess t-CyCIF CRC1 3D atlas feature tables into a single h5ad.

Source: mytrain/data/cycif_crc1_3d/feature_tables/WD-76845-XXX-features.zip
(25 slices of the CRC1 specimen, Lin et al., Cell 2023).

Output: mytrain/data/cycif_crc1_3d/processed/cycif_crc1.h5ad
  - X:    raw single-cell mean intensities of the 24-plex marker panel (float32)
  - obs:  slice_id (int), slice (category), AREA, CIRC
  - obsm['spatial']: (Xt, Yt, z_um) — registered full-image pixel coordinates
          plus z in um computed from section number and thickness
          (5 um for sections <= 84, 4 um from section 85 onward).
  - obsm['spatial_um']: all three axes in um — x/y converted from pixels at
          0.65 um/pixel (PhysicalSizeX/Y in the official OME-TIFF metadata of
          e.g. data/WD-76845-002.ome.tif; 20x 0.75 NA, CyteFinder).

Usage:
    conda activate ntf
    python mytrain/scripts/preprocess_cycif_crc1.py
"""

import os
import re
import zipfile

import anndata
import numpy as np
import pandas as pd

DATA_DIR = '/home/gongyuqiao/ur_annotation/NTF/mytrain/data/cycif_crc1_3d/feature_tables'
OUT_PATH = '/home/gongyuqiao/ur_annotation/NTF/mytrain/data/cycif_crc1_3d/processed/cycif_crc1.h5ad'

PIXEL_SIZE_UM = 0.65  # from OME-TIFF PhysicalSizeX/Y metadata

MARKERS = ['CD3', 'Ki67', 'CD4', 'CD20', 'CD163', 'Ecadherin', 'LaminABC',
           'PCNA', 'NaKATPase', 'Keratin', 'CD45', 'CD68', 'FOXP3', 'Vimentin',
           'Desmin', 'Ki67_570', 'CD45RO', 'aSMA', 'PD1', 'CD8a', 'PDL1',
           'CDX2', 'CD31', 'Collagen']


def slice_z_um(section):
    """z position in um: 5 um sections up to #84, 4 um sections from #85."""
    return section * 5.0 if section <= 84 else 84 * 5.0 + (section - 84) * 4.0


def main():
    zips = sorted(f for f in os.listdir(DATA_DIR) if f.endswith('-features.zip'))
    print(f'{len(zips)} feature tables found')

    obs_parts, x_parts, xyz_parts = [], [], []
    for z in zips:
        section = int(re.search(r'WD-76845-(\d+)-features\.zip', z).group(1))
        with zipfile.ZipFile(os.path.join(DATA_DIR, z)) as zf:
            with zf.open(zf.namelist()[0]) as f:
                df = pd.read_csv(f, usecols=MARKERS + ['AREA', 'CIRC', 'Xt', 'Yt'])
        n = len(df)
        xyz = np.column_stack([
            df['Xt'].to_numpy(np.float32),
            df['Yt'].to_numpy(np.float32),
            np.full(n, slice_z_um(section), dtype=np.float32),
        ])
        obs = pd.DataFrame({
            'slice_id': np.full(n, section, dtype=np.int32),
            'AREA': df['AREA'].to_numpy(np.float32),
            'CIRC': df['CIRC'].to_numpy(np.float32),
        })
        obs_parts.append(obs)
        x_parts.append(df[MARKERS].to_numpy(np.float32))
        xyz_parts.append(xyz)
        print(f'  slice {section:3d}: {n:,} cells, z={slice_z_um(section):.0f} um')

    obs = pd.concat(obs_parts, ignore_index=True)
    obs['slice'] = pd.Categorical(obs['slice_id'].astype(str))
    obs.index = [f'cell_{i}' for i in range(len(obs))]
    X = np.vstack(x_parts)
    xyz = np.vstack(xyz_parts)

    adata = anndata.AnnData(
        X=X,
        obs=obs,
        var=pd.DataFrame(index=MARKERS),
    )
    adata.obsm['spatial'] = xyz
    xyz_um = xyz.copy()
    xyz_um[:, 0] *= PIXEL_SIZE_UM
    xyz_um[:, 1] *= PIXEL_SIZE_UM
    adata.obsm['spatial_um'] = xyz_um
    adata.uns['dataset'] = 't-CyCIF CRC1 3D atlas (Lin et al., Cell 2023)'
    adata.uns['pixel_size_um'] = PIXEL_SIZE_UM
    adata.uns['z_note'] = ('z in um from section number: 5 um sections up to #84, '
                           '4 um sections from #85; x/y in registered pixels '
                           '(spatial) or um at 0.65 um/pixel (spatial_um)')

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    adata.write_h5ad(OUT_PATH, compression='gzip')
    print(f'Wrote {OUT_PATH}: {adata.n_obs:,} cells x {adata.n_vars} markers')


if __name__ == '__main__':
    main()
