# ntf/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReconstructionLoss(torch.nn.Module):
    """
    Negative log-likelihood loss assuming a multivariate Gaussian with a
    spherical covariance matrix (i.e., a single shared variance per spot).
    """
    def __init__(self):
        super().__init__()

    def forward(self, g_true, g_bar, g_var):
        """
        Calculates the loss.
        Args:
            g_true (torch.Tensor): Ground truth expression, shape (N, n_genes).
            g_bar (torch.Tensor): Predicted mean expression, shape (N, n_genes).
            g_var (torch.Tensor): Predicted SHARED variance, shape (N, 1).
        """
        n_genes = g_true.shape[1]
        
        # Ensure g_var is squeezed to (N,) for calculation
        g_var_squeezed = g_var.squeeze()

        # log(det(Sigma)) term, where Sigma = g_var * I
        # det(Sigma) = (g_var)^n_genes, so log(det(Sigma)) = n_genes * log(g_var)
        term1 = n_genes * torch.log(g_var_squeezed)

        # (g - mu)^T * Sigma^-1 * (g - mu) term
        # This becomes (1/g_var) * sum_of_squared_errors
        sum_sq_err = ((g_true - g_bar).pow(2)).sum(dim=1)
        term2 = sum_sq_err / g_var_squeezed
        
        loss = 0.5 * (term1 + term2)
        return loss.mean()
    
# class ReconstructionLoss(torch.nn.Module):
#     """
#     Negative log-likelihood loss for a diagonal multivariate Gaussian.
#     """
#     def __init__(self):
#         super().__init__()

#     def forward(self, g_true, g_bar, g_var):
#         term1 = torch.log(g_var).sum(dim=1)
#         term2 = ((g_true - g_bar).pow(2) / g_var).sum(dim=1)
#         loss = 0.5 * (term1 + term2)
#         return loss.mean()

# class ReconstructionLoss(torch.nn.Module):

#     def __init__(self):
#         super().__init__()

#     def forward(self, g_true, g_bar, g_var):
#         mse_loss = nn.MSELoss()
#         return mse_loss(g_bar, g_true)
    
class SmoothnessLoss(torch.nn.Module):
    """
    Enforces smoothness by penalizing differences between predictions
    at nearby points.
    """
    def __init__(self, noise_std: float = 0.1):
        super().__init__()
        self.noise_std = noise_std
    
    def forward(self, model, coords, slice_ids):
        # Create perturbed coordinates
        noise = torch.randn_like(coords) * self.noise_std
        perturbed_coords = coords + noise

        # Get predictions for both original and perturbed coordinates
        preds_orig = model(coords, slice_ids)
        preds_pert = model(perturbed_coords, slice_ids)

        # Calculate MSE for density and conditional gene expression
        loss_v = F.mse_loss(preds_orig["v"], preds_pert["v"])
        loss_g = F.mse_loss(preds_orig["g_hat"], preds_pert["g_hat"])
        
        return loss_v, loss_g