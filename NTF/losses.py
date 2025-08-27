# ntf/losses.py
import torch
import torch.nn.functional as F

class ReconstructionLoss(torch.nn.Module):
    """
    Negative log-likelihood loss for a diagonal multivariate Gaussian.
    """
    def __init__(self):
        super().__init__()

    def forward(self, g_true, g_bar, g_var):
        term1 = torch.log(g_var).sum(dim=1)
        term2 = ((g_true - g_bar).pow(2) / g_var).sum(dim=1)
        loss = 0.5 * (term1 + term2)
        return loss.mean()

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