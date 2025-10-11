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
    # 如果启用了dropout预测，则也为概率创建一个张量
    dropout_probs_tensor = None
    if not model.args.no_dropout:
        dropout_probs_tensor = torch.empty(xyz_flat.shape[0], model.n_genes, dtype=torch.float32, device=xyz.device)
    
    
    with torch.no_grad():
        for i in range(0, xyz_flat.shape[0], batch_size):
            s = slice(i, i + batch_size)
            xyz_batch = xyz_flat[s]
            
            # --- Inference with Dropout Logic ---
            # 1. 获取模型原始输出
            results = model(xyz_batch[:, None])
            
            # 2. 获取预测的表达值
            v_pred = results["expression"].squeeze(1) # 移除n_samples维度
            
            if not model.args.no_dropout:
                # 3. 获取dropout概率
                dropout_prob = results["dropout_prob"].squeeze(1)
                dropout_probs_tensor[s] = dropout_prob
                
                # 4. 大于阈值则进行dropout
                dropout_mask = dropout_prob < 0.5  # 这里的0.5是一个经验阈值，可以根据需要调整
                
                # 5. 应用掩码
                v_batch = v_pred * dropout_mask
            else:
                v_batch = v_pred

            dropout_prob = results["dropout_prob"].squeeze(1)
            dropout_probs_tensor[s] = dropout_prob
            # --- End Inference Logic ---
                
            v[s] = v_batch
            # v[s] = v_pred

        output = {"expression": v.view(*shape, model.n_genes)}
        if dropout_probs_tensor is not None:
            output["dropout_prob"] = dropout_probs_tensor.view(*shape, model.n_genes)
            
    return output