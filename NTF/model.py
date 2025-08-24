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
    Neural Transcriptome Field (NTF) Model.
    This model maps a 3D coordinate to a G+1-dimensional output (G gene expression + 1 density).
    """
    def __init__(self, num_genes: int, config: dict):
        """
        Initializes the NTF model.

        Parameters:
            num_genes (int): The total number of genes in the dataset (G).
            config (dict): A dictionary containing the encoder and MLP configuration.
        """
        super().__init__()
        if tcnn is None:
            raise ImportError("tiny-cuda-nn is required to run this model.")

        self.num_genes = num_genes
        
        # define multi-resolution hashing encoder
        self.encoder = tcnn.Encoding(
            n_input_dims=3,  # input is 3D coordinates (x, y, z)
            encoding_config=config["encoder"]
        )
        
        # define MLP for gene expression and density prediction
        self.mlp = tcnn.Network(
            n_input_dims=self.encoder.n_output_dims, # input dimension is the output of the encoder
            n_output_dims=self.num_genes + 1,        # output dimension is G (gene expressions) + 1 (density)
            network_config=config["network"]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.

        Parameters:
            x (torch.Tensor): Input 3D coordinate tensor, shape (N, 3).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Gene expression profiles and densities.
                - gene_expressions (torch.Tensor): shape (N, num_genes)
                - density (torch.Tensor): shape (N, 1)
        """
        # 1. encode the input coordinates
        features = self.encoder(x)
        
        # 2. predict gene expressions and density using MLP
        # output of tiny-cuda-nn is in float16, convert to float32 for consistency
        outputs = self.mlp(features).to(torch.float32)
        
        # 3. seperate gene expressions and density
        gene_expressions = outputs[..., :self.num_genes]
        density = outputs[..., self.num_genes:]
        
        # 4. activate the outputs
        gene_expressions = torch.relu(gene_expressions) 
        density = torch.relu(density)
        
        return gene_expressions, density