# dataset.py
import torch
import anndata
import numpy as np
from typing import Union
import pandas as pd
from torch.utils.data import Dataset

class SpatialOmicsDataset(Dataset):
    """
    A PyTorch Dataset to handle AnnData objects for spatial omics.

    Args:
        adata (anndata.AnnData): The AnnData object containing the data.
            It must have .obsm['spatial'], .obs[slice_id_key], and .X.
    """
    def __init__(self, adata: anndata.AnnData, slice_id_key: str = 'slice_id'):
        super().__init__()
        
        # 1. Extract coordinates
        self.coords = torch.from_numpy(adata.obsm['spatial']).float()
        
        # 2. Extract and encode slice IDs
        if not pd.api.types.is_categorical_dtype(adata.obs[slice_id_key]):
            adata.obs[slice_id_key] = adata.obs[slice_id_key].astype('category')
        self.slice_ids = torch.from_numpy(adata.obs['slice_id'].cat.codes.to_numpy().copy()).long()
        
        # 3. Extract gene expression
        if hasattr(adata.X, "toarray"):
            self.genes = torch.from_numpy(adata.X.toarray()).float()
        else:
            self.genes = torch.from_numpy(adata.X).float()
            
        self.n_spots = adata.n_obs
        self.n_genes = adata.n_vars
        self.n_slices = len(adata.obs[slice_id_key].cat.categories)

    def __len__(self):
        return self.n_spots

    def __getitem__(self, idx):
        return {
            "coords": self.coords[idx],
            "slice_id": self.slice_ids[idx],
            "genes": self.genes[idx]
        }