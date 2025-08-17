# dataset.py
import torch
import anndata
import numpy as np

class SpatialOmicsDataset(torch.utils.data.Dataset):
    """
    用于空间组学数据的PyTorch Dataset.
    假设输入数据是经过配准的 AnnData 对象，其中 Z 坐标已添加.
    """
    def __init__(self, adata_path: str):
        adata = anndata.read_h5ad(adata_path)
        
        # 提取坐标 (需要包含Z轴信息)
        if 'spatial_3d' not in adata.obsm:
            raise ValueError("AnnData object must have 'spatial_3d' in .obsm after registration.")
        self.coords = torch.from_numpy(adata.obsm['spatial_3d'].astype(np.float32))
        
        # 提取基因表达
        self.expressions = torch.from_numpy(adata.X.toarray().astype(np.float32)) # 确保是稠密矩阵
        
        # 计算数据边界，用于损失函数采样
        self.bounds = (self.coords.min(dim=0).values, self.coords.max(dim=0).values)

    def __len__(self):
        return self.coords.shape[0]

    def __getitem__(self, idx):
        return {
            "origin": self.coords[idx],
            "target_genes": self.expressions[idx],
        }