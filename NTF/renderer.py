# renderer.py
import torch.nn as nn
import torch
from .model import NeuralTranscriptomicField
from typing import Union, Tuple, List

class Renderer:
    """
    Handles the Monte Carlo rendering process to simulate the acquisition model.
    """
    def __init__(self, model: NeuralTranscriptomicField, psf_std: Union[float, Tuple[float, float, float], List[float]] = 1.0):
        self.model = model

        # Handle isotropic vs. anisotropic PSF standard deviation
        if isinstance(psf_std, (float, int)):
            psf_std_tensor = torch.tensor([psf_std, psf_std, psf_std], dtype=torch.float32)
        elif isinstance(psf_std, (tuple, list)) and len(psf_std) == 3:
            psf_std_tensor = torch.tensor(psf_std, dtype=torch.float32)
        else:
            raise ValueError("psf_std must be a float or a list/tuple of 3 floats.")
            
        # Reshape for broadcasting: (1, 1, 3)
        self.psf_std_tensor = psf_std_tensor.view(1, 1, 3)

    def render_spots(self, spot_coords: torch.Tensor, slice_ids: torch.Tensor, K: int = 16):
        """
        Renders the expected gene expression for a batch of spots using MC sampling.
        
        Args:
            spot_coords (torch.Tensor): Spot coordinates, shape (N, 3).
            slice_ids (torch.Tensor): Slice IDs for each spot, shape (N,).
            K (int): Number of Monte Carlo samples per spot.
            
        Returns:
            tuple: Rendered mean (g_bar) and variance (g_var) of gene expression.
        """
        batch_size = spot_coords.shape[0]
        device = spot_coords.device
        self.psf_std_tensor = self.psf_std_tensor.to(device)

        # 1. Monte Carlo Sampling to simulate PSF
        noise = torch.randn(batch_size, K, 3, device=device) * self.psf_std_tensor
        sampled_coords = spot_coords.unsqueeze(1) + noise # (N, K, 3)
        flat_sampled_coords = sampled_coords.reshape(-1, 3) # (N * K, 3)

        # 2. Repeat slice_ids for each sample
        flat_slice_ids = slice_ids.unsqueeze(1).repeat(1, K).reshape(-1) # (N * K,)
        
        # 3. Get predictions from the model at sampled points
        preds = self.model(flat_sampled_coords, flat_slice_ids)
        
        # 4. Aggregate samples to get rendered spot expression
        # 4a. Reshape predictions
        v_samples = preds["v"].view(batch_size, K, 1)
        g_hat_samples = preds["g_hat"].view(batch_size, K, -1)
        b_samples = preds["b"].view(batch_size, K, -1)
        # sigma2_samples = preds["sigma2"].view(batch_size, K, -1)

        # 4b. Apply acquisition model
        total_g_samples = v_samples * g_hat_samples
        biased_g_samples = b_samples * total_g_samples
        g_bar = torch.mean(biased_g_samples, dim=1) # (N, n_genes)


        sigma2_samples = preds["sigma2"].view(batch_size, K, 1) # Shape is now (N, K, 1)
        # Average the scalar variances from all samples for that spot
        # We simplify here by not having the bias field affect the variance
        intrinsic_spot_var = torch.mean(sigma2_samples, dim=1) # Shape becomes (N, 1)
        
        
        # biased_sigma2_samples = b_samples.pow(2) * sigma2_samples
        # g_var = torch.mean(biased_sigma2_samples, dim=1) # (N, n_genes)

        # 4c. Apply slice scaling factor
        slice_scales_all = torch.softmax(self.model.slice_scaling_unconstrained, dim=0) * self.model.slice_embeddings.num_embeddings
        slice_scales = slice_scales_all[slice_ids]
        
        g_bar = slice_scales.unsqueeze(1) * g_bar
        g_var = slice_scales.unsqueeze(1).pow(2) * intrinsic_spot_var + 1e-6 # Add epsilon for stability

        return g_bar, g_var