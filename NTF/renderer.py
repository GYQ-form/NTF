# renderer.py
import torch

def sample_points_along_rays(
    origins: torch.Tensor, 
    slice_thickness: float, 
    num_samples: int
) -> torch.Tensor:
    """
    沿着垂直于XY平面的光线进行采样.

    参数:
        origins (torch.Tensor): spot的中心坐标 (N, 3).
        slice_thickness (float): 切片厚度.
        num_samples (int): 每条光线的采样点数.

    返回:
        torch.Tensor: 采样点坐标, shape (N, num_samples, 3).
    """
    N = origins.shape[0]
    t_vals = torch.linspace(0.0, 1.0, steps=num_samples, device=origins.device)
    z_offsets = (t_vals - 0.5) * slice_thickness
    
    sample_points = origins.unsqueeze(1).expand(-1, num_samples, -1).clone()
    sample_points[..., 2] += z_offsets.unsqueeze(0).T
    
    return sample_points, z_offsets

def volume_render(
    model: nn.Module,
    origins: torch.Tensor,
    slice_thickness: float,
    num_samples: int
) -> torch.Tensor:
    """
    对一组光线进行体渲染，计算最终的基因表达谱.

    参数:
        model (nn.Module): NTF模型.
        origins (torch.Tensor): spot的中心坐标 (N, 3).
        slice_thickness (float): 切片厚度.
        num_samples (int): 每条光线的采样点数.

    返回:
        torch.Tensor: 渲染出的基因表达向量, shape (N, num_genes).
    """
    # 1. 沿光线采样点
    points, z_vals = sample_points_along_rays(origins, slice_thickness, num_samples)
    # points shape: (N, num_samples, 3)
    
    # 2. 查询模型获取每个采样点的基因表达和密度
    points_flat = points.view(-1, 3)
    g_vals, sigma_vals = model(points_flat)
    
    g_vals = g_vals.view(points.shape[0], points.shape[1], -1)
    sigma_vals = sigma_vals.view(points.shape[0], points.shape[1])
    
    # 3. 计算alpha合成的权重
    dists = z_vals[1:] - z_vals[:-1]
    dists = torch.cat([dists, torch.tensor([1e10], device=origins.device).expand(dists[0].shape)], 0)
    dists = dists.T * slice_thickness

    alpha = 1.0 - torch.exp(-sigma_vals * dists)
    
    # 计算透射率 T_i = product(1 - alpha_{j}) for j < i
    transmittance = torch.cumprod(torch.cat([
        torch.ones(alpha.shape[0], 1, device=origins.device), 
        1. - alpha + 1e-10
    ], -1), -1)[:, :-1]
    
    weights = alpha * transmittance
    
    # 4. 积分得到最终的基因表达向量
    rendered_genes = torch.sum(weights.unsqueeze(-1) * g_vals, 1)
    
    return rendered_genes