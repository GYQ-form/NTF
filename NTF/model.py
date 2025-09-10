# model.py
import torch
import torch.nn as nn
try:
    import tinycudann as tcnn
except ImportError:
    print("tiny-cuda-nn is not installed. This model requires it.")
    tcnn = None

class GenePredictor(nn.Module):
    """
    Gene Predictor: Transformer + MLP.
    Takes latent features z(x) and predicts conditional gene expression G_hat(x).
    """
    def __init__(self, latent_dim: int, n_genes: int, n_heads: int = 4, hidden_dim: int = 128):
        super().__init__()
        # single-layer Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # MLP head
        self.mlp_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_genes)
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Transformer expect input size (N, S, E)，where S is sequence length
        # here we set S as 1
        z_float = z.to(torch.float32)
        if z_float.ndim == 2:
            z_float = z_float.unsqueeze(1) # (batch * K, 1, latent_dim)
            
        transformer_out = self.transformer_encoder(z_float).squeeze(1) # (batch * K, latent_dim)
        
        g_hat = self.mlp_head(transformer_out)
        return g_hat
    
    
class NeuralTranscriptomicField(nn.Module):
    """
    Neural Transcriptomics Field (NTF) Model.
    This model takes 3D coordinates and slice IDs, and outputs the predicted
    cell density, conditional gene expression, bias, and variance at those points.
    The Monte Carlo rendering is handled by the Renderer class.
    """
    def __init__(self,
                 n_genes: int,
                 n_slices: int,
                 latent_dim: int = 32,
                 embedding_dim: int = 16,
                 log2_hashmap_size: int = 19,
                 n_levels: int = 16,
                 n_features_per_level: int = 2,
                 n_bias_levels: int = 4):
        super().__init__()
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.n_bias_levels = n_bias_levels
        self.n_features_per_level = n_features_per_level

        # --- Slice-specific Trainable Parameters ---
        self.slice_embeddings = nn.Embedding(n_slices, embedding_dim)
        self.slice_scaling_unconstrained = nn.Parameter(torch.zeros(n_slices))

        # --- Shared Coordinate Encoder ---
        self.shared_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": n_features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": 16,
                "per_level_scale": 2.0,
            }
        )

        # --- Prediction Networks ---
        # a. MLP_VZ: predicts cell density V(x) and latent features z(x)
        self.net_vz = tcnn.Network(
            n_input_dims=self.shared_encoder.n_output_dims, 
            n_output_dims=latent_dim + 1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            },
        )

        # b. gene predictor (Transformer + MLP)
        self.gene_predictor = tcnn.Network(
            n_input_dims=latent_dim,
            n_output_dims=n_genes,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "ReLU",
                "n_neurons": 64,
                "n_hidden_layers": 2, # 1 hidden layer + 1 output layer = 2-layer MLP
            }
        )

        # c. MLP_B: predicts bias field B_i(x)
        # input: low-freq encodings + slice embedding -> output: B_i(x) (n_genes)
        n_low_freq_features = n_features_per_level * n_bias_levels
        self.net_b = tcnn.Network(
            n_input_dims=n_low_freq_features + embedding_dim,
            n_output_dims=n_genes,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Softplus",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )
        
        # d. MLP_sigma: predicts noise variance sigma_i^2(x)
        # input: z(x) + slice embedding -> output: sigma_i^2(x) (n_genes)
        # self.net_sigma = tcnn.Network(
        #     n_input_dims=latent_dim + embedding_dim,
        #     n_output_dims=n_genes,
        #     network_config={
        #         "otype": "FullyFusedMLP",
        #         "activation": "ReLU",
        #         "output_activation": "Softplus",
        #         "n_neurons": 64,
        #         "n_hidden_layers": 1,
        #     },
        # )

        # Instead of a tcnn.Network, we use nn.Sequential to get fine-grained control.
        self.net_sigma = nn.Sequential(
            nn.Linear(latent_dim + embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus() # The final activation is still Softplus
        )
        
        # Apply the custom initialization to the final linear layer of net_sigma
        # We target the second Linear layer, which is at index -2.
        with torch.no_grad():
            self.net_sigma[-2].bias.fill_(0)
            # You can also initialize weights for better practice
            nn.init.kaiming_normal_(self.net_sigma[-2].weight, mode='fan_in', nonlinearity='relu')       

    def forward(self, coords: torch.Tensor, slice_ids: torch.Tensor, K: int = 64):
        """
        Forward pass of the model. Predicts values at exact coordinates.
        
        Args:
            coords (torch.Tensor): Shape (N, 3)
            slice_ids (torch.Tensor): Shape (N,)
            
        Returns:
            dict: A dictionary with predictions for v, g_hat, b, and sigma2.
        """

        # 1. Encode coordinates
        all_level_encodings = self.shared_encoder(coords)
        
        # 2. Get slice-specific embeddings
        slice_embs = self.slice_embeddings(slice_ids)
        
        # 3. Predict V(x) and z(x)
        vz_output = self.net_vz(all_level_encodings)
        v = torch.nn.functional.softplus(vz_output[:, 0:1])
        z = vz_output[:, 1:]
        
        # 4. Predict G_hat(x)
        g_hat = self.gene_predictor(z)
        
        # 5. Predict B_i(x)
        n_low_freq_features = self.n_features_per_level * self.n_bias_levels
        low_freq_enc = all_level_encodings[:, :n_low_freq_features]
        bias_input = torch.cat([low_freq_enc, slice_embs], dim=1).contiguous()
        b = self.net_b(bias_input)
        
        # 6. Predict sigma_i^2(x)
        sigma_input = torch.cat([z.detach(), slice_embs], dim=1).contiguous()
        sigma2 = self.net_sigma(sigma_input)

        return {"v": v, "g_hat": g_hat, "b": b, "sigma2": sigma2}