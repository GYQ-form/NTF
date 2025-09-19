from typing import Dict
import torch
import anndata
from scipy.spatial import cKDTree
import numpy as np

class SpatialOmicsDataset:
    """
    Creates a dataset from an AnnData object for training the NTF model.
    """
    def __init__(self, adata: anndata.AnnData, slice_id_obs: str = 'slice_id', spatial_obsm: str = 'spatial'):
        
        if slice_id_obs not in adata.obs:
            raise ValueError(f"Slice ID column '{slice_id_obs}' not found in adata.obs")
        if spatial_obsm not in adata.obsm:
            raise ValueError(f"Spatial coordinates '{spatial_obsm}' not found in adata.obsm")

        self.xyz = torch.from_numpy(adata.obsm[spatial_obsm]).float()
        self.v = torch.from_numpy(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X).float()
        
        # Create slice indices from obs column
        slice_ids = adata.obs[slice_id_obs].astype('category').cat
        self.slice_idx = torch.from_numpy(slice_ids.codes.to_numpy().copy()).long()
        self.slice_map = dict(enumerate(slice_ids.categories))
        self.n_slices = len(self.slice_map)
        
        # adaptive resolution per slice estimation
        self.resolution = self._estimate_resolution(self.xyz)

        self.count = 0
        self.epoch = 0
        self.n_genes = self.v.shape[1]

    @staticmethod
    def _estimate_resolution(xyz: torch.Tensor, n_samples: int = 2000) -> torch.Tensor:
        """
        高效估计分辨率：随机采样点做最近邻统计，适用于百万级数据。
        支持 torch.Tensor 或 np.ndarray 输入。
        返回 shape=[3] 的每轴分辨率估计。
        """
        xyz_np = xyz.cpu().numpy() if isinstance(xyz, torch.Tensor) else xyz
        N = xyz_np.shape[0]
        n_samples = min(n_samples, N)
        idx = np.random.choice(N, n_samples, replace=False)
        xyz_sample = xyz_np[idx]
        tree = cKDTree(xyz_np)
        dists, _ = tree.query(xyz_sample, k=2)
        nn_dists = dists[:, 1]
        # 各轴分开估计
        per_axis_median = []
        for axis in range(3):
            diffs = xyz_sample[None, :, axis] - xyz_np[:, None, axis]
            abs_diffs = np.abs(diffs)
            abs_diffs[abs_diffs == 0] = np.inf
            per_axis_median.append(np.median(np.min(abs_diffs, axis=0)))
        return torch.tensor(per_axis_median, dtype=torch.float32)
        
    @property
    def bounding_box(self) -> torch.Tensor:
        # Add a margin to the bounding box
        margin = self.resolution.max() * 2
        xyz_min = self.xyz.amin(0) - margin
        xyz_max = self.xyz.amax(0) + margin
        return torch.stack([xyz_min, xyz_max], 0)

    @property
    def expression_mean(self) -> torch.Tensor:
        return self.v.mean(0)

    def get_batch(self, batch_size: int, device) -> Dict[str, torch.Tensor]:
        if self.count + batch_size > self.xyz.shape[0]:  # New epoch
            self.count = 0
            self.epoch += 1
            # Shuffle data
            idx = torch.randperm(self.xyz.shape[0])
            self.xyz = self.xyz[idx]
            self.v = self.v[idx]
            self.slice_idx = self.slice_idx[idx]
        
        # Fetch a batch and move to device
        s = slice(self.count, self.count + batch_size)
        batch = {
            "xyz": self.xyz[s].to(device),
            "v": self.v[s].to(device),
            "slice_idx": self.slice_idx[s].to(device),
        }
        self.count += batch_size
        return batch

    # The mask property is kept for API consistency with NeSVoR's workflow,
    # but might be less critical if you only sample at specific points.
    @property
    def mask(self) -> torch.Tensor:
        # This creates a low-resolution volume mask of the tissue space
        with torch.no_grad():
            resolution_min = self.resolution.min()
            xyz_min = self.xyz.amin(0)
            xyz_max = self.xyz.amax(0)
            shape_xyz = ((xyz_max - xyz_min) / resolution_min).ceil().long()
            shape = (int(shape_xyz[2]), int(shape_xyz[1]), int(shape_xyz[0]))
            
            kji = ((self.xyz - xyz_min) / resolution_min).round().long()
            
            mask = torch.bincount(
                kji[..., 0] + shape[2] * kji[..., 1] + shape[2] * shape[1] * kji[..., 2],
                minlength=shape[0] * shape[1] * shape[2],
            ).view(shape) > 0
            
            return mask.float()