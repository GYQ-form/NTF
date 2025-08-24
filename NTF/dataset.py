# dataset.py
import torch
import anndata
import numpy as np
from typing import Union

class SpatialOmicsDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for Spatial Omics Data.
    We assume the input is AnnData object, Z axis has been added to adata.obsm['spatial_3d'].

    Parameters
    ----------
    adata_input : Union[str, anndata.AnnData]
        Either a path to an h5ad file or an AnnData object directly
    """
    def __init__(self, adata_input: Union[str, anndata.AnnData]):
        # Handle input - can be either path or AnnData object
        if isinstance(adata_input, str):
            adata = anndata.read_h5ad(adata_input)
        elif isinstance(adata_input, anndata.AnnData):
            adata = adata_input
        else:
            raise TypeError("Input must be either a path (str) or an AnnData object")
        
        # load spatial coordinates
        if 'spatial_3d' not in adata.obsm:
            raise ValueError("AnnData object must have 'spatial_3d' in .obsm after registration.")
        self.coords = torch.from_numpy(adata.obsm['spatial_3d'].astype(np.float32))
        
        # extract gene expression matrix
        if not isinstance(adata.X, np.ndarray):
            if hasattr(adata.X, "toarray"):
                self.expressions = torch.from_numpy(adata.X.toarray().astype(np.float32))
            else:
                raise ValueError("adata.X is not a numpy array or sparse matrix with toarray() method.")
        else:
            self.expressions = torch.from_numpy(adata.X.astype(np.float32))
        
        # calculate bounds
        self.bounds = (self.coords.min(dim=0).values, self.coords.max(dim=0).values)

    def __len__(self):
        return self.coords.shape[0]

    def __getitem__(self, idx):
        return {
            "origin": self.coords[idx],
            "target_genes": self.expressions[idx],
        }