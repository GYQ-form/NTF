import torch
from .models import GeneINR

def sample_points(
    model: GeneINR,
    xyz: torch.Tensor,
    batch_size: int = 4096,
) -> torch.Tensor:
    """
    Sample gene expression at given 3D coordinates.
    """
    model.eval()
    shape = xyz.shape[:-1]
    xyz_flat = xyz.view(-1, 3)
    
    # Pre-allocate tensor for results
    v = torch.empty(xyz_flat.shape[0], model.n_genes, dtype=torch.float32, device=xyz.device)
    
    with torch.no_grad():
        for i in range(0, xyz_flat.shape[0], batch_size):
            xyz_batch = xyz_flat[i : i + batch_size]
            v_batch = model(xyz_batch) # Directly call INR
            v[i : i + batch_size] = v_batch
            
    return v.view(*shape, model.n_genes)