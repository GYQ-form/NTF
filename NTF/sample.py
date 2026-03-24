import torch
from .models import GeneINR

def sample_points(
    model: GeneINR,
    xyz: torch.Tensor,
    batch_size: int = 4096,
) -> torch.Tensor:
    """
    Sample gene expression at given 3D coordinates, considering dropout prediction.
    """
    model.eval()
    shape = xyz.shape[:-1]
    xyz_flat = xyz.view(-1, 3)
    
    v = torch.empty(xyz_flat.shape[0], model.n_genes, dtype=torch.float32, device=xyz.device)
    # If dropout prediction is enabled, also allocate a tensor for probabilities
    dropout_probs_tensor = None
    if not model.args.no_dropout:
        dropout_probs_tensor = torch.empty(xyz_flat.shape[0], model.n_genes, dtype=torch.float32, device=xyz.device)
    
    
    with torch.no_grad():
        for i in range(0, xyz_flat.shape[0], batch_size):
            s = slice(i, i + batch_size)
            xyz_batch = xyz_flat[s]
            
            # --- Inference with Dropout Logic ---
            # 1. Get raw model output
            results = model(xyz_batch[:, None])
            
            # 2. Get predicted expression values
            v_pred = results[0].squeeze(1) # remove n_samples dimension
            
            if not model.args.no_dropout:
                # 3. Get dropout probabilities
                dropout_logits = results[1].squeeze(1)
                dropout_prob = torch.sigmoid(dropout_logits)
                dropout_probs_tensor[s] = dropout_prob
                
                # 4. Apply dropout where probability exceeds threshold
                dropout_mask = dropout_prob < 0.5  # 0.5 is an empirical threshold, adjust as needed
                
                # 5. Apply mask
                v_batch = v_pred * dropout_mask
            else:
                v_batch = v_pred
            # --- End Inference Logic ---
                
            v[s] = v_batch
            # v[s] = v_pred

        output = {"expression": v.view(*shape, model.n_genes)}
        if dropout_probs_tensor is not None:
            output["dropout_prob"] = dropout_probs_tensor.view(*shape, model.n_genes)
            
    return output