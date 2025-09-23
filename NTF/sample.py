import torch
from .models import GeneINR, NeuralTranscriptomicField # 导入主模型

def sample_points(
    model: NeuralTranscriptomicField, # <-- 注意：现在传入主模型，而不是inr
    xyz: torch.Tensor,
    slice_idx: torch.Tensor, # <-- 新增：需要slice_id来获取embedding
    batch_size: int = 4096,
) -> torch.Tensor:
    """
    Sample gene expression at given 3D coordinates, considering dropout prediction.
    """
    model.eval()
    shape = xyz.shape[:-1]
    xyz_flat = xyz.view(-1, 3)
    slice_idx_flat = slice_idx.view(-1)
    
    v = torch.empty(xyz_flat.shape[0], model.n_genes, dtype=torch.float32, device=xyz.device)
    # 如果启用了dropout预测，则也为概率创建一个张量
    dropout_probs_tensor = None
    if not model.args.no_dropout:
        dropout_probs_tensor = torch.empty(xyz_flat.shape[0], model.n_genes, dtype=torch.float32, device=xyz.device)
    
    
    with torch.no_grad():
        for i in range(0, xyz_flat.shape[0], batch_size):
            s = slice(i, i + batch_size)
            xyz_batch = xyz_flat[s]
            slice_idx_batch = slice_idx_flat[s]
            
            # --- Inference with Dropout Logic ---
            # 1. 获取模型原始输出
            se_batch = model.slice_embedding(slice_idx_batch) if model.args.n_features_slice > 0 else None
            results = model.net_forward(xyz_batch[:, None], se_batch[:, None] if se_batch is not None else None)
            
            # 2. 获取预测的表达值
            v_pred = results["expression"].squeeze(1) # 移除n_samples维度
            
            if not model.args.no_dropout:
                # 3. 获取dropout概率
                dropout_logits = results["dropout_logits"].squeeze(1)
                dropout_prob = torch.sigmoid(dropout_logits)
                dropout_probs_tensor[s] = dropout_prob
                
                # 4. 从伯努利分布采样
                dropout_mask = torch.bernoulli(1 - dropout_prob) # 1-prob: 不发生dropout的概率
                
                # 5. 应用掩码
                v_batch = v_pred * dropout_mask
            else:
                v_batch = v_pred
            # --- End Inference Logic ---
                
            v[i : i + batch_size] = v_batch

        output = {"expression": v.view(*shape, model.n_genes)}
        if dropout_probs_tensor is not None:
            output["dropout_prob"] = dropout_probs_tensor.view(*shape, model.n_genes)
            
    return output