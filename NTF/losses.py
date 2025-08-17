# losses.py
import torch

def reconstruction_loss(predicted_genes, target_genes, loss_type='l2'):
    """计算重建损失."""
    if loss_type == 'l1':
        return torch.nn.functional.l1_loss(predicted_genes, target_genes)
    else:
        return torch.nn.functional.mse_loss(predicted_genes, target_genes)

def geometric_loss(model, bounds, num_points=1024*16):
    """计算几何正则化损失，鼓励密度场平滑."""
    min_bound, max_bound = bounds
    points = torch.rand(num_points, 3, device=min_bound.device) * (max_bound - min_bound) + min_bound
    points.requires_grad_(True)
    
    _, density = model(points)
    
    # 计算密度梯度
    grad_outputs = torch.ones_like(density)
    grad = torch.autograd.grad(
        outputs=density,
        inputs=points,
        grad_outputs=grad_outputs,
        create_graph=True
    )[0]
    
    # 惩罚梯度的L2范数
    return grad.norm(2, dim=-1).mean()

def smoothness_loss(model, bounds, num_points=1024*16, epsilon=0.01):
    """计算空间平滑度损失，鼓励邻近点的基因表达相似."""
    min_bound, max_bound = bounds
    points1 = torch.rand(num_points, 3, device=min_bound.device) * (max_bound - min_bound) + min_bound
    
    # 在邻近区域随机采样第二个点
    noise = (torch.rand_like(points1) * 2 - 1) * epsilon
    points2 = torch.clamp(points1 + noise, min_bound, max_bound)
    
    g1, _ = model(points1)
    g2, _ = model(points2)
    
    return torch.nn.functional.mse_loss(g1, g2)