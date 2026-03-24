import argparse
import os
import torch
import sys
import logging

def add_shared_args(description="NTF Model Training"):
    """
    定义并解析所有命令行参数。
    
    参数:
    - description: 命令行帮助信息的描述文本。
    
    返回:
    - args: 解析后的参数命名空间对象。
    """
    parser = argparse.ArgumentParser(description=description, add_help=False)

    # ==========================================
    #           1. 核心 I/O 参数
    # ==========================================
    io_group = parser.add_argument_group('Input/Output')
    io_group.add_argument('--input_data','-i', type=str, required=True, help='Path to your .h5ad file. If not provided, mock data will be used.')
    io_group.add_argument('--output_dir', '-o', type=str, required=True, help='Directory to save results. If None, auto-generated.')
    io_group.add_argument('--log_dir', type=str, default='runs', help='Directory for TensorBoard logs.')
    io_group.add_argument('--slice_id', type=str, default='brain_section_label', help='Column in adata.obs indicating slice IDs.')
    
    # ==========================================
    #           2. 训练参数 (Training)
    # ==========================================
    train_group = parser.add_argument_group('Training Parameters')
    train_group.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Device to use for training.')
    train_group.add_argument('--n_epochs', type=int, default=None, help='Total training epochs.')
    train_group.add_argument('--n_iter', type=int, default=20000, help='Total training iterations.')
    train_group.add_argument('--batch_size', type=int, default=8192, help='Batch size for training.')
    train_group.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate.')
    train_group.add_argument('--milestones', type=float, nargs='+', default=[0.5, 0.7, 0.9], help='LR scheduler milestones (as fractions of n_iter).')
    train_group.add_argument('--gamma', type=float, default=0.5, help='LR decay factor.')


    # ==========================================
    #           3. 模型结构参数 (Model Architecture)
    # ==========================================
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('--base_resolution', type=int, default=2)
    model_group.add_argument('--n_levels', type=float, default=12)
    model_group.add_argument('--level_scale', type=float, default=1.5)
    model_group.add_argument('--n_features_per_level', type=int, default=2)
    model_group.add_argument('--log2_hashmap_size', type=int, default=19)
    model_group.add_argument('--width', type=int, default=64, help='Width of MLP layers.')
    model_group.add_argument('--depth', type=int, default=1, help='Depth of MLP layers.')
    model_group.add_argument('--n_features_slice', type=int, default=16, help='Dimension of slice embeddings.')
    model_group.add_argument('--n_features_z', type=int, default=16, help='Dimension of latent features for variance net.')
    model_group.add_argument('--reg_neighbor_radius', type=float, default=0.01, help="Radius for sampling neighbor points for regularization (in normalized space).")
    model_group.add_argument('--no_pixel_variance', action='store_true', help='Disable per-pixel variance prediction.')
    model_group.add_argument('--no_slice_variance', action='store_true', help='Disable per-slice variance prediction.')
    model_group.add_argument('--n_levels_bias', type=int, default=0, help='Levels for bias network.')
    
    # ==========================================
    #           4. 损失与正则化 (Loss & Regularization)
    # ==========================================
    loss_group = parser.add_argument_group('Loss and Regularization')
    loss_group.add_argument('--weight_expr', type=float, default=0.1, help='Weight for expression regularization.')
    loss_group.add_argument('--weight_bias', type=float, default=0.1)
    loss_group.add_argument('--single_precision', action='store_true', help='Use single precision (fp32) instead of mixed precision.')
    loss_group.add_argument('--n_samples', type=int, default=4, help='Number of samples per point for PSF simulation.')
    loss_group.add_argument('--early_stopping_patience', type=int, default=5, help='Patience for early stopping.')
    loss_group.add_argument('--early_stopping_delta', type=float, default=1e-4, help='Min delta for early stopping.')
    loss_group.add_argument('--early_stopping_check_interval', type=int, default=200, help='Iteration interval for early stopping check.')
    
    # Dropout 相关
    loss_group.add_argument('--no_dropout', action='store_true', help='Disable the dropout prediction network to handle zero-inflation.')
    loss_group.add_argument('--weight_dropout', type=float, default=2000.0, help='Weight for dropout loss.')
    
    return parser


def process_args(args):
    """
    对解析后的参数进行通用的后处理（设置设备、数据类型、创建目录）。
    """
    # 1. 数据类型
    args.dtype = torch.float32 if args.single_precision else torch.float16

    # 2. 自动设备选择
    if args.device == 'auto':
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # 4. 确保目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    return args