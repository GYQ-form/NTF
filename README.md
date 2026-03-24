# Neural Transcriptomic Field (NTF)

NTF is a deep learning framework for reconstructing continuous 3D gene expression fields from sparse spatial transcriptomics data. It uses a hash-grid-based implicit neural representation (INR) to model gene expression as a continuous function of 3D coordinates, enabling high-resolution reconstruction across multiple tissue sections.

## Model Architecture

![NTF Model Architecture](https://github.com/user-attachments/assets/99e34cce-9b46-4e1c-b5e0-a0c9c49bdd0b)

The NTF model consists of the following key components:

- **Hash Grid Encoding**: Multi-resolution hash grid that encodes 3D spatial coordinates into compact feature vectors.
- **Expression Network (GeneINR)**: An MLP that maps encoded features to gene expression values using a softplus activation.
- **Dropout Network**: An auxiliary MLP that predicts dropout probabilities to handle zero-inflation in spatial transcriptomics data.
- **Slice Embedding**: Learnable embeddings for each tissue section to capture section-specific effects.
- **Variance Network**: Predicts per-pixel uncertainty (variance) for probabilistic modeling.
- **Bias Network** (optional): Predicts section-level bias corrections.

## Installation

```bash
pip install -e .
```

For GPU acceleration with [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn):
```bash
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

If `tinycudann` is not available, NTF will automatically fall back to a pure PyTorch implementation.

## Requirements

See `requirements.txt` for the full list of dependencies. Key dependencies:
- `torch`
- `anndata`
- `scipy`
- `tqdm`
- `tensorboard`

## Usage

### Python API

```python
import anndata
from NTF import train, sample_points
from NTF.config import add_shared_args, process_args

# Load your spatial transcriptomics data (AnnData format)
adata = anndata.read_h5ad("your_data.h5ad")

# Parse and process arguments
parser = add_shared_args()
args = parser.parse_args([
    "--input_data", "your_data.h5ad",
    "--output_dir", "./output",
    "--slice_id", "slice_id",   # obs column with section labels
])
args = process_args(args)

# Train the model
model = train(adata, args)

# Sample gene expression at 3D coordinates
import torch
xyz = torch.tensor([[x, y, z], ...], dtype=torch.float32)
output = sample_points(model, xyz)
expression = output["expression"]  # shape: [N, n_genes]
```

### Command-line Arguments

| Argument | Default | Description |
|---|---|---|
| `--input_data` / `-i` | required | Path to `.h5ad` input file |
| `--output_dir` / `-o` | required | Directory to save results |
| `--slice_id` | `brain_section_label` | `adata.obs` column for slice IDs |
| `--n_iter` | 20000 | Total training iterations |
| `--batch_size` | 8192 | Batch size |
| `--learning_rate` | 1e-4 | Learning rate |
| `--width` | 64 | MLP layer width |
| `--depth` | 1 | MLP depth |
| `--n_levels` | 12 | Number of hash grid levels |
| `--n_features_per_level` | 2 | Features per hash grid level |
| `--no_dropout` | False | Disable dropout prediction network |
| `--weight_expr` | 0.1 | Expression smoothness regularization weight |
| `--single_precision` | False | Use fp32 instead of mixed precision |

## Input Data Format

The model expects an `AnnData` object with:
- `adata.X`: Gene expression matrix (cells × genes), can be sparse
- `adata.obsm['spatial']`: 3D spatial coordinates (cells × 3)
- `adata.obs[slice_id]`: Section/slice identifier column

## License

See [LICENSE](LICENSE) for details.
