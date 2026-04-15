# NTF Scripts

This directory contains command-line scripts built on top of the NTF library. Every script shares a common set of arguments defined in `NTF/config.py` and adds task-specific arguments on top.

---

## Shared Arguments

All scripts accept the following common arguments (defined in `NTF.config.add_shared_args`):

### Input / Output

| Argument | Default | Description |
|---|---|---|
| `--input_data` / `-i` | **required** | Path to the `.h5ad` input file |
| `--output_dir` / `-o` | **required** | Directory to save results |
| `--log_dir` | `runs` | Directory for TensorBoard logs |
| `--slice_id` | `brain_section_label` | `adata.obs` column containing slice/section IDs |

### Training

| Argument | Default | Description |
|---|---|---|
| `--device` | `cuda:0` / `cpu` | Device for training |
| `--n_iter` | 20000 | Total number of training iterations |
| `--n_epochs` | `None` | Alternative to `--n_iter`; overrides it when set |
| `--batch_size` | 8192 | Mini-batch size |
| `--learning_rate` | 1e-4 | Learning rate |
| `--milestones` | `0.5 0.7 0.9` | LR scheduler milestones (as fractions of `n_iter`) |
| `--gamma` | 0.5 | LR decay factor at each milestone |

### Model Architecture

| Argument | Default | Description |
|---|---|---|
| `--base_resolution` | 2 | Base hash grid resolution |
| `--n_levels` | 12 | Number of hash grid levels |
| `--level_scale` | 1.5 | Scale factor between grid levels |
| `--n_features_per_level` | 2 | Features per hash grid level |
| `--log2_hashmap_size` | 19 | Log2 of the hash map size |
| `--width` | 64 | Width of MLP hidden layers |
| `--depth` | 1 | Depth of MLP |
| `--n_features_slice` | 16 | Dimension of per-slice learnable embeddings |
| `--n_features_z` | 16 | Dimension of latent features for the variance network |
| `--n_levels_bias` | 0 | Number of hash grid levels for the bias network |
| `--no_pixel_variance` | False | Disable per-pixel variance prediction |
| `--no_slice_variance` | False | Disable per-slice variance prediction |
| `--reg_neighbor_radius` | 0.01 | Radius for neighbor sampling in expression regularization |

### Loss & Regularization

| Argument | Default | Description |
|---|---|---|
| `--weight_expr` | 0.1 | Weight for expression smoothness regularization |
| `--weight_bias` | 0.1 | Weight for bias regularization |
| `--n_samples` | 4 | Samples per point for PSF simulation |
| `--single_precision` | False | Use fp32 instead of mixed precision (fp16) |
| `--no_dropout` | False | Disable the dropout prediction network |
| `--weight_dropout` | 2000.0 | Weight for the dropout binary cross-entropy loss |
| `--early_stopping_patience` | 5 | Early stopping patience (in check intervals) |
| `--early_stopping_delta` | 1e-4 | Minimum loss improvement to reset patience |
| `--early_stopping_check_interval` | 200 | Iteration interval between early stopping checks |

---

## Scripts

### `train_and_eval.py`

Train an NTF model and evaluate reconstruction quality on a held-out test set.

#### Task-Specific Arguments

| Argument | Default | Description |
|---|---|---|
| `--split_mode` | `random` | Split strategy: `random` or `label` |
| `--test_size` | 0.1 | Fraction of data held out for testing (used when `--split_mode=random`) |
| `--split_column` | `None` | `adata.obs` column used to determine the split (required when `--split_mode=label`) |
| `--train_label` | `None` | Value in `--split_column` that marks training cells (required when `--split_mode=label`) |
| `--test_label` | `None` | Value in `--split_column` that marks test cells (required when `--split_mode=label`) |

#### Example

```bash
# Random 90/10 split
python scripts/train_and_eval.py \
    -i data.h5ad \
    -o results/ \
    --n_iter 20000

# Label-based split using a pre-defined column
python scripts/train_and_eval.py \
    -i data.h5ad \
    -o results/ \
    --split_mode label \
    --split_column group \
    --train_label train \
    --test_label test
```

#### Outputs

- `results/test_results.h5ad` – AnnData with predicted expression in `layers['prediction']`
- `results/gene_metrics.csv` – Per-gene Pearson and Spearman correlation
- `results/res.log` – Summary log with inference time and average correlations

---

### `generate_hires_convex.py`

Train a model and generate a high-resolution point cloud by sampling new 3D coordinates inside the **convex hull** of the original tissue. Best suited for roughly convex tissues.

#### Task-Specific Arguments

| Argument | Default | Description |
|---|---|---|
| `--n_generate` | 1000000 | Number of new points to generate |
| `--cat_keys` | `[]` | Categorical variables in `adata.obs` to predict via KNN voting (e.g., `cell_type`) |
| `--k_neighbors` | 5 | Number of nearest neighbours for KNN majority voting |

#### Example

```bash
python scripts/generate_hires_convex.py \
    -i data.h5ad \
    -o hires_output/ \
    --n_generate 500000 \
    --cat_keys cell_type
```

#### Output

- `hires_output/50w.h5ad` – High-resolution AnnData with predicted expression and optional categorical annotations

---

### `generate_hires_distfield.py`

Train a model and generate a high-resolution point cloud by accepting candidate points whose nearest-neighbour distance to the original tissue is within a configurable tolerance. Better suited for **non-convex** or complex tissue shapes.

#### Task-Specific Arguments

| Argument | Default | Description |
|---|---|---|
| `--n_generate` | 1000000 | Number of new points to generate |
| `--dist_tol` | 20 | Tolerance multiplier: a candidate point is accepted if its distance to the nearest original cell is less than `avg_inter-cell-distance × dist_tol` |
| `--expr_filter_ratio` | 0.03 | Remove generated points whose total expression is below `max_total_expr × expr_filter_ratio` (set to 0 to disable) |
| `--cat_keys` | `[]` | Categorical variables to predict via KNN voting |
| `--k_neighbors` | 5 | Number of nearest neighbours for KNN majority voting |

#### Example

```bash
python scripts/generate_hires_distfield.py \
    -i data.h5ad \
    -o hires_output/ \
    --n_generate 500000 \
    --dist_tol 15 \
    --expr_filter_ratio 0.05 \
    --cat_keys cell_type
```

#### Output

- `hires_output/50w.h5ad` – High-resolution AnnData with predicted expression and optional categorical annotations

---

### `sectioning_data_prepare.py`

Train a model and pre-compute all assets required by the interactive sectioning application.

#### Task-Specific Arguments

| Argument | Default | Description |
|---|---|---|
| `--alpha_multiplier` | 14.0 | Multiplier for the average inter-cell distance when computing the Alpha Shape mesh |
| `--cat_keys` | `[]` | Categorical variables in `adata.obs` to store for KNN-based prediction in the app |

#### Example

```bash
python scripts/sectioning_data_prepare.py \
    -i data.h5ad \
    -o sectioning_assets/ \
    --cat_keys cell_type region
```

#### Outputs

- `sectioning_assets/trained_inr.pt` – Serialised trained INR model
- `sectioning_assets/metadata.pkl` – Mesh vertices/faces, raw coordinates, spatial scaling, gene names, and categorical variable encodings

---

### `sectioning_app.py`

Interactive **Dash** web application for in-silico virtual sectioning. Loads pre-computed assets from `sectioning_data_prepare.py` and provides a real-time UI to:

- Visualise the 3D tissue mesh
- Define a cutting plane by specifying a normal vector and a point on the plane
- Run NTF inference on the 2D slice
- Display per-gene expression or categorical annotations on the slice
- Download the slice as an `.h5ad` file

#### Setup

Edit the `ASSETS_DIR` variable at the top of `sectioning_app.py` to point to the directory produced by `sectioning_data_prepare.py`:

```python
ASSETS_DIR = '/path/to/sectioning_assets'
```

#### Launch

```bash
python scripts/sectioning_app.py
```

The app will be available at `http://localhost:8201` by default.

#### Dependencies

In addition to the core NTF requirements, this script requires:
- `dash`
- `dash-bootstrap-components`
- `plotly`
- `open3d` (required by `sectioning_data_prepare.py` for mesh generation)

---

### `3D_visulization.py`

Interactive **Dash** web application for exploring `.h5ad` spatial transcriptomics data in 3D. Supports both categorical variable visualisation and multi-gene expression overlay with advanced rendering options.

#### Task-Specific Arguments

| Argument | Default | Description |
|---|---|---|
| `--input_path` / `-i` | **required** | Path to the `.h5ad` input file |
| `--obs-cols` | `cell_type leiden cluster domain slice_id …` | `adata.obs` columns to expose as categorical visualisation options |
| `--port` | 8060 | Port to serve the application on |
| `--base-url` | `/` | URL path prefix (useful when running behind a reverse proxy) |
| `--title` | `3D Spatial Transcriptomics Visualization` | Title displayed in the app header |

#### Example

```bash
python scripts/3D_visulization.py \
    -i data.h5ad \
    --obs-cols cell_type leiden \
    --port 8060
```

The app will be available at `http://localhost:8060` by default.

#### Features

- **Categorical view**: colour points by any `adata.obs` column (e.g., `cell_type`, `leiden`)
- **Gene expression view**: visualise up to 5 genes simultaneously with independent colour scales
- Expression filtering: hide zero-expression cells, set a minimum expression threshold, and optionally filter by a reference gene
- **Adaptive opacity**: expression-proportional transparency for cleaner multi-gene overlays
- Controls for point size and global opacity
- Collapsible sidebar; immersive mode hides axis backgrounds
- Automatic down-sampling to 100 000 cells when the dataset exceeds this threshold

#### Dependencies

In addition to the core NTF requirements, this script requires:
- `dash`
- `plotly`
- `scanpy`
