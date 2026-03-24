from argparse import Namespace
from math import log2
from typing import Optional, Dict, Any, Union, Tuple
import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
from .hash_grid_torch import HashEmbedder
from .utils import resolution2sigma

# --- tcnn integration ---
USE_TORCH = False
try:
    import tinycudann as tcnn
except ImportError:
    logging.warning("Fail to load tinycudann. Will use pure PyTorch implementation.")
    USE_TORCH = True
except Exception as e:
    logging.warning(f"An error occurred during tinycudann import: {e}. Will use pure PyTorch implementation.")
    USE_TORCH = True
# ------------------------

# key for loss/regularization
D_LOSS = "MSE"
S_LOSS = "logVar"
DS_LOSS = "MSE+logVar"
DO_LOSS = "DropoutBCE"
B_REG = "biasReg"
E_REG = "exprReg"

def build_encoding(**config):
    if USE_TORCH:
        # Pop tinycudann-specific params that HashEmbedder doesn't use
        config.pop("dtype", None)
        return HashEmbedder(**config)
    else:
        n_input_dims = config.pop("n_input_dims")
        dtype = config.pop("dtype")
        try:
            return tcnn.Encoding(n_input_dims=n_input_dims, encoding_config=config, dtype=dtype)
        except RuntimeError as e:
            if "TCNN was not compiled with half-precision support" in str(e):
                logging.error("TCNN was not compiled with half-precision support! Try using --single-precision.")
            raise e

def build_network(**config):
    dtype = config.pop("dtype")
    if dtype == torch.float16 and not USE_TORCH:
        return tcnn.Network(
            n_input_dims=config["n_input_dims"],
            n_output_dims=config["n_output_dims"],
            network_config={
                "otype": "CutlassMLP",
                "activation": config["activation"],
                "output_activation": config["output_activation"],
                "n_neurons": config["n_neurons"],
                "n_hidden_layers": config["n_hidden_layers"],
            },
        )
    else:
        activation = getattr(nn, config["activation"]) if config["activation"] != "None" else None
        output_activation = getattr(nn, config["output_activation"]) if config["output_activation"] != "None" else None
        models = []
        if config["n_hidden_layers"] > 0:
            models.append(nn.Linear(config["n_input_dims"], config["n_neurons"]))
            for _ in range(config["n_hidden_layers"] - 1):
                if activation: models.append(activation())
                models.append(nn.Linear(config["n_neurons"], config["n_neurons"]))
            if activation: models.append(activation())
            models.append(nn.Linear(config["n_neurons"], config["n_output_dims"]))
        else:
            models.append(nn.Linear(config["n_input_dims"], config["n_output_dims"]))
        if output_activation: models.append(output_activation())
        return nn.Sequential(*models)


class GeneINR(nn.Module):
    def __init__(
        self,
        n_genes: int,
        bounding_box: torch.Tensor,
        args: Namespace,
    ) -> None:
        super().__init__()
        self.register_buffer("bounding_box", bounding_box)
        self.n_genes = n_genes
        self.args = args
        
        base_resolution = getattr(args, 'base_resolution', 2)
        n_levels = getattr(args, 'n_levels', 8)

        self.encoding = build_encoding(
            n_input_dims=3, otype="HashGrid", n_levels=n_levels,
            n_features_per_level=args.n_features_per_level,
            log2_hashmap_size=args.log2_hashmap_size,
            base_resolution=base_resolution, per_level_scale=args.level_scale,
            dtype=args.dtype,
        )
        
        self.expression_net = build_network(
            n_input_dims=self.encoding.n_output_dims,
            n_output_dims=self.n_genes + args.n_features_z,
            activation="ReLU", output_activation="None",
            n_neurons=args.width, n_hidden_layers=args.depth,
            dtype=args.dtype,
        )

        if not args.no_dropout:
            self.dropout_net = build_network(
                n_input_dims=self.encoding.n_output_dims,
                n_output_dims=n_genes,
                activation="ReLU",
                output_activation="None",
                n_neurons=args.width,
                n_hidden_layers=args.depth,
                dtype=args.dtype,
            )

        logging.debug(
            "hyperparameters for hash grid encoding: "
            + "lowest_grid_size=%d, highest_grid_size=%d, scale=%1.2f, n_levels=%d",
            base_resolution,
            int(base_resolution * args.level_scale ** (n_levels - 1)),
            args.level_scale,
            n_levels,
        )

    def forward(self, x: torch.Tensor):
        x_norm = (x - self.bounding_box[0]) / (self.bounding_box[1] - self.bounding_box[0])
        prefix_shape = x_norm.shape[:-1]
        x_flat = x_norm.view(-1, 3)
        
        pe = self.encoding(x_flat)
        if not self.training and not USE_TORCH:
            pe = pe.to(dtype=x.dtype) # Match precision for inference
        
        z = self.expression_net(pe)
        z = z.view(*prefix_shape, -1)
        if not self.args.no_dropout:
            do_logits = self.dropout_net(pe)
            do_logits = do_logits.view(*prefix_shape, -1)
        else:
            do_logits = None

        expression = F.softplus(z[..., :self.n_genes])
        

        if self.training:
            return expression, do_logits, pe, z
        else:
            return expression, do_logits
        # return expression, pe, z


    def sample_batch(
        self,
        xyz: torch.Tensor,
        psf_sigma: Union[float, torch.Tensor],
        n_samples: int,
    ) -> torch.Tensor:
        if n_samples > 1:
            if isinstance(psf_sigma, torch.Tensor):
                psf_sigma = psf_sigma.view(-1, 1, 3)
            xyz_psf = torch.randn(
                xyz.shape[0], n_samples, 3, dtype=xyz.dtype, device=xyz.device
            )
            xyz = xyz[:, None] + xyz_psf * psf_sigma
        else:
            xyz = xyz[:, None]
        return xyz


class NeuralTranscriptomicField(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_slices: int,
        resolution: torch.Tensor,
        bounding_box: torch.Tensor,
        args: Namespace,
        pos_weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        global USE_TORCH
        if "cpu" in str(args.device):
            USE_TORCH = True
        
        self.args = args
        self.n_slices = n_slices
        self.n_genes = n_genes
        # Use a simplified resolution sigma, assuming uniform for all slices for now
        self.psf_sigma = resolution2sigma(resolution, isotropic=False)
        self.pos_weight = pos_weight
        self.build_network(bounding_box)
        self.to(args.device)

    def build_network(self, bounding_box) -> None:
        if self.args.n_features_slice > 0:
            self.slice_embedding = nn.Embedding(
                self.n_slices, self.args.n_features_slice
            )
        
        if not self.args.no_slice_variance:
            self.log_var_slice = nn.Parameter(
                torch.zeros(self.n_slices, self.n_genes, dtype=torch.float32)
            )
            
        self.inr = GeneINR(self.n_genes, bounding_box, self.args)
        
        
        if not self.args.no_pixel_variance:
            self.sigma_net = build_network(
                n_input_dims=self.args.n_features_slice + self.args.n_features_z,
                n_output_dims=self.n_genes, activation="ReLU", output_activation="None",
                n_neurons=self.args.width, n_hidden_layers=1, dtype=self.args.dtype,
            )
        

        if self.args.n_levels_bias > 0:
            n_encoding_dims = self.args.n_levels_bias * self.args.n_features_per_level
            self.b_net = build_network(
                n_input_dims=n_encoding_dims + self.args.n_features_slice,
                n_output_dims=self.n_genes, activation="ReLU", output_activation="None",
                n_neurons=self.args.width, n_hidden_layers=1, dtype=self.args.dtype,
            )

    def forward(
        self,
        xyz: torch.Tensor,
        v: torch.Tensor, # gene expression vector
        slice_idx: torch.Tensor,
    ) -> Dict[str, Any]:
        
        n_samples = self.args.n_samples
        psf_sigma_batch = self.psf_sigma.to(xyz.device)
        xyz_sampled = self.inr.sample_batch(xyz, psf_sigma_batch, n_samples)
        se = self.slice_embedding(slice_idx)[:, None].expand(-1, n_samples, -1) if self.args.n_features_slice > 0 else None
            
        results = self.net_forward(xyz_sampled, se)
        
        v_out_samples = results["expression"]
        log_bias = results.get("log_bias", torch.tensor(0))
        bias = torch.exp(log_bias)
        v_out = (bias * v_out_samples).mean(1)
        var = torch.exp(results.get("log_var", 0))
        if not self.args.no_slice_variance:
            var = var + self.log_var_slice.exp()[slice_idx].unsqueeze(1)
        var = (bias.detach()**2 * var).mean(1)

        losses = {}
        if not self.args.no_dropout:
            # 1. Compute dropout loss
            target_is_zero = (v == 0).float() # target: whether the true expression is zero
            dropout_logits = results["dropout_logits"].mean(1) # average over sampled points
            loss_do = F.binary_cross_entropy_with_logits(dropout_logits, target_is_zero, pos_weight=self.pos_weight.to(v.device))
            losses[DO_LOSS] = loss_do

            # 2. Compute reconstruction loss only on non-zero values
            non_zero_mask = (v > 0)
            if non_zero_mask.sum() > 0:
                loss_d = ((v_out[non_zero_mask] - v[non_zero_mask]) ** 2 / (2 * var[non_zero_mask])).mean()
                loss_s = 0.5 * var[non_zero_mask].log().mean()
            else: # if the entire batch is zero, loss is 0
                loss_d = torch.tensor(0.0, device=v.device)
                loss_s = torch.tensor(0.0, device=v.device)
            losses[D_LOSS] = loss_d
            losses[S_LOSS] = loss_s
            losses[DS_LOSS] = loss_d + loss_s
        else:
            # Standard reconstruction loss
            loss_d = ((v_out - v) ** 2 / (2 * var)).mean()
            loss_s = 0.5 * var.log().mean()
            losses[D_LOSS] = loss_d
            losses[S_LOSS] = loss_s
            losses[DS_LOSS] = loss_d + loss_s


        if self.args.n_levels_bias > 0:
            losses[B_REG] = log_bias.mean() ** 2
            
        losses[E_REG] = self.expression_reg(v_out_samples, xyz_sampled)

        return losses

    def net_forward(
        self,
        x: torch.Tensor,
        se: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        expression, do_logits, pe, z = self.inr(x)
        prefix_shape = expression.shape[:-1]
        results = {"expression": expression, "dropout_logits": do_logits}

        zs = []
        if se is not None:
            # tcnn network expects flat input
            zs.append(se.reshape(-1, se.shape[-1]))

        if self.args.n_levels_bias > 0:
            pe_bias = pe[..., : self.args.n_levels_bias * self.args.n_features_per_level]
            bias_input = torch.cat(zs + [pe_bias], -1)
            results["log_bias"] = self.b_net(bias_input).view(*prefix_shape, self.n_genes)

        if not self.args.no_pixel_variance:
            z_var = z.view(-1, z.shape[-1])[..., self.n_genes:]
            var_input = torch.cat(zs + [z_var], -1)
            results["log_var"] = self.sigma_net(var_input).view(*prefix_shape, self.n_genes)

        if not self.training:
            results["z"] = z

        return results

    def expression_reg(self, expression, xyz):
        """
        Compute smoothness regularization loss.
        Selects different strategies based on the value of args.weight_expr.
        """
        if self.args.weight_expr == 0:
            return 0.0

        # 1. Use a subset of sampled points for regularization to reduce computation
        n_sample = min(4, expression.shape[1])
        xyz_sub = xyz[:, :n_sample].flatten(0, 1)
        expr_sub = expression[:, :n_sample].flatten(0, 1)

        # 2. Create "neighbor points" around each point by adding a small random perturbation
        #    within the range [-radius, +radius]
        radius = self.args.reg_neighbor_radius
        with torch.no_grad(): # no gradient needed for creating neighbor points
            # Generate a random direction vector in [-1, 1]
            perturbation = (torch.rand_like(xyz_sub) * 2 - 1) 
            # Scale to the specified radius
            perturbation *= radius
            xyz_neighbor = xyz_sub + perturbation

        # 3. Compute gene expression predictions at neighbor points
        #    A single standard forward pass avoids double-backward issues
        neighbor_expression = self.inr(xyz_neighbor)[0] if self.inr.training else self.inr(xyz_neighbor)

        # 4. Compute MSE between original and neighbor expression as the regularization loss,
        #    encouraging smoothness in the predicted expression field
        loss = torch.mean((expr_sub - neighbor_expression) ** 2)
        return loss
