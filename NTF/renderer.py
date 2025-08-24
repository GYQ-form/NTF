# renderer.py
import torch.nn as nn
import torch

def sample_points_along_rays(
    origins: torch.Tensor, 
    slice_thickness: float, 
    num_samples: int
) -> torch.Tensor:
    """
    Sampling along rays perpendicular to the XY plane.

    Parameters:
        origins (torch.Tensor): The center coordinates of the spot (N, 3).
        slice_thickness (float): The slice thickness.
        num_samples (int): The number of sampling points per ray.

    Returns:
        torch.Tensor: The sampling point coordinates, shape (N, num_samples, 3).
    """
    N = origins.shape[0]
    t_vals = torch.linspace(0.0, 1.0, steps=num_samples, device=origins.device)
    z_offsets = (t_vals - 0.5) * slice_thickness
    
    sample_points = origins.unsqueeze(1).expand(-1, num_samples, -1).clone()
    sample_points[..., 2] += z_offsets.unsqueeze(0)
    
    return sample_points, z_offsets

def volume_render(
    model: nn.Module,
    origins: torch.Tensor,
    slice_thickness: float,
    num_samples: int
) -> torch.Tensor:
    """
    Perform volume rendering on a set of rays and calculate the final gene expression profile.

    Parameters:
        model (nn.Module): NTF model.
        origins (torch.Tensor): The center coordinates of the spot (N, 3).
        slice_thickness (float): The slice thickness.
        num_samples (int): The number of samples per ray.

    Returns:
        torch.Tensor: The rendered gene expression vector, shape (N, num_genes).
    """
    # 1. sample points along rays
    points, z_vals = sample_points_along_rays(origins, slice_thickness, num_samples) # points shape: (N, num_samples, 3)
    
    # 2. inquire the model for gene expression and sigma values
    points_flat = points.view(-1, 3)
    g_vals, sigma_vals = model(points_flat)
    
    g_vals = g_vals.view(points.shape[0], points.shape[1], -1)
    sigma_vals = sigma_vals.view(points.shape[0], points.shape[1])
    
    # 3. 以批处理方式正确计算距离和alpha权重
    # 获取每个采样点的z坐标 (shape: N, num_samples)
    z_coords = points[..., 2]
    
    # 计算相邻采样点之间的距离 (shape: N, num_samples - 1)
    dists = z_coords[:, 1:] - z_coords[:, :-1]
    
    # 为最后一个采样区间创建一个非常大的距离 (shape: N, 1)
    # This correctly creates a value for each ray in the batch.
    infinity_dist = torch.full((dists.shape[0], 1), 1e10, device=origins.device)
    
    # 将所有距离拼接在一起 (shape: N, num_samples)
    dists = torch.cat([dists, infinity_dist], dim=-1)
    
    # 计算alpha值, alpha = 1 - exp(-sigma * delta)
    # 这里的 dists 和 sigma_vals 形状都是 (N, num_samples)，可以安全地逐元素相乘
    alpha = 1.0 - torch.exp(-sigma_vals * dists)
    
    # calculate transmittance: T_i = product(1 - alpha_{j}) for j < i
    transmittance = torch.cumprod(torch.cat([
        torch.ones(alpha.shape[0], 1, device=origins.device), 
        1. - alpha + 1e-10
    ], -1), -1)[:, :-1]
    
    weights = alpha * transmittance
    
    # 4. integrate the gene values
    rendered_genes = torch.sum(weights.unsqueeze(-1) * g_vals, 1)
    
    return rendered_genes