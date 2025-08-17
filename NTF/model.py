# model.py
import torch
import torch.nn as nn
try:
    import tinycudann as tcnn
except ImportError:
    print("tiny-cuda-nn is not installed. This model requires it.")
    tcnn = None

class NeuralTranscriptomicField(nn.Module):
    """
    神经转录组场 (NTF) 模型.
    
    该模型将一个3D坐标映射到一个G+1维的输出 (G个基因表达 + 1个密度).
    """
    def __init__(self, num_genes: int, config: dict):
        """
        初始化NTF模型.
        
        参数:
            num_genes (int): 数据集中的基因总数 (G).
            config (dict): 包含编码器和MLP配置的字典.
        """
        super().__init__()
        if tcnn is None:
            raise ImportError("tiny-cuda-nn is required to run this model.")

        self.num_genes = num_genes
        
        # 定义多分辨率哈希编码器
        self.encoder = tcnn.Encoding(
            n_input_dims=3,  # 输入是3D坐标 (x, y, z)
            encoding_config=config["encoder"]
        )
        
        # 定义用于预测基因和密度的小型MLP
        self.mlp = tcnn.Network(
            n_input_dims=self.encoder.n_output_dims, # 输入维度是编码器的输出维度
            n_output_dims=self.num_genes + 1,        # 输出维度是 基因数 + 1 (密度)
            network_config=config["network"]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        模型的前向传播.

        参数:
            x (torch.Tensor): 输入的3D坐标张量, shape为 (N, 3).

        返回:
            tuple[torch.Tensor, torch.Tensor]: 基因表达谱和密度.
                - gene_expressions (torch.Tensor): shape (N, num_genes)
                - density (torch.Tensor): shape (N, 1)
        """
        # 1. 对输入坐标进行编码
        features = self.encoder(x)
        
        # 2. 通过MLP进行预测
        # tiny-cuda-nn的MLP输出是FP16，需要转为FP32
        outputs = self.mlp(features).to(torch.float32)
        
        # 3. 分离输出为基因表达和密度
        gene_expressions = outputs[..., :self.num_genes]
        density = outputs[..., self.num_genes:]
        
        # 4. 应用激活函数
        # 基因表达通常是非负稀疏的，ReLU或Softplus是好的选择
        gene_expressions = torch.relu(gene_expressions) 
        # 密度必须为非负
        density = torch.relu(density)
        
        return gene_expressions, density