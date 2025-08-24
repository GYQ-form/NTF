# losses.py
import torch

# def reconstruction_loss(predicted_genes, target_genes, loss_type='l2'):
#     """calculate reconstruction loss between predicted and target gene expressions."""
#     if loss_type == 'l1':
#         return torch.nn.functional.l1_loss(predicted_genes, target_genes)
#     else:
#         return torch.nn.functional.mse_loss(predicted_genes, target_genes)

def reconstruction_loss(predicted_genes, target_genes, positive_weight=10):
    """
    计算重建损失.
    新增 positive_weight 参数以解决稀疏性问题.
    
    参数:
        predicted_genes (torch.Tensor): 模型的预测值.
        target_genes (torch.Tensor): 真实的基因表达值.
        positive_weight (float): 对非零真实值的损失所乘的权重.
    """
    # 计算每个元素的误差 (例如, L2误差)
    error = torch.pow((predicted_genes - target_genes),2)
    
    # 创建一个与target_genes形状相同的权重张量
    # 真实值 > 0 的位置权重为 positive_weight, 否则为 1.0
    weights = torch.ones_like(target_genes)
    weights[target_genes > 0] = positive_weight
    
    # 将误差与权重相乘
    weighted_error = error * weights
    
    # 返回加权误差的均值
    return weighted_error.mean()

def geometric_loss(model, bounds, num_points=1024*16, epsilon=0.01):
    """
    Encourage smooth density fields by penalizing density differences between neighboring points
    """
    min_bound, max_bound = bounds

    # 1. randomly sample points within the bounds
    points = torch.rand(num_points, 3, device=min_bound.device) * (max_bound - min_bound) + min_bound
    
    # 2. perturb points slightly
    noise = (torch.rand_like(points) * 2 - 1) * epsilon
    perturbed_points = torch.clamp(points + noise, min_bound, max_bound)

    _, density_orig = model(points)
    _, density_perturbed = model(perturbed_points)

    return torch.nn.functional.mse_loss(density_orig, density_perturbed)


def smoothness_loss(model, bounds, num_points=1024*16, epsilon=0.01):
    """calculate smoothness loss, ensuring neighboring points have similar outputs."""
    min_bound, max_bound = bounds
    points1 = torch.rand(num_points, 3, device=min_bound.device) * (max_bound - min_bound) + min_bound
    
    # sample points with small perturbations
    noise = (torch.rand_like(points1) * 2 - 1) * epsilon
    points2 = torch.clamp(points1 + noise, min_bound, max_bound)
    
    g1, _ = model(points1)
    g2, _ = model(points2)
    
    return torch.nn.functional.mse_loss(g1, g2)