import torch
import anndata
import numpy as np
import logging
import argparse
import os
import time
import matplotlib
matplotlib.use('Agg')  # 支持无界面服务器
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
# 导入我们创建的包
from NTF.train import train
from NTF.sample import sample_points
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from NTF.helper_func import calculate_spatial_metrics

def get_args():
    parser = argparse.ArgumentParser(description="Train and evaluate NeuroTField model.")
    parser.add_argument('--input_data','-i', type=str, required=True, help='Path to your .h5ad file. If not provided, mock data will be used.')
    parser.add_argument('--output_dir', '-o', type=str, required=True, help='Directory to save results. If None, auto-generated.')
    parser.add_argument('--slice_id', type=str, default='brain_section_label', help='Column in adata.obs indicating slice IDs.')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Device to use for training.')
    parser.add_argument('--n_epochs', type=int, default=None, help='Total training epochs.')
    parser.add_argument('--n_iter', type=int, default=20000, help='Total training iterations.')
    parser.add_argument('--batch_size', type=int, default=8192, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--milestones', type=float, nargs='+', default=[0.5, 0.7, 0.9], help='LR scheduler milestones (as fractions of n_iter).')
    parser.add_argument('--gamma', type=float, default=0.5, help='LR decay factor.')
    parser.add_argument('--base_resolution', type=int, default=2)
    parser.add_argument('--n_levels', type=float, default=12)
    parser.add_argument('--level_scale', type=float, default=1.5)
    parser.add_argument('--n_features_per_level', type=int, default=2)
    parser.add_argument('--log2_hashmap_size', type=int, default=19)
    parser.add_argument('--width', type=int, default=64, help='Width of MLP layers.')
    parser.add_argument('--depth', type=int, default=1, help='Depth of MLP layers.')
    parser.add_argument('--n_features_slice', type=int, default=16, help='Dimension of slice embeddings.')
    parser.add_argument('--n_features_z', type=int, default=16, help='Dimension of latent features for variance net.')
    parser.add_argument('--weight_expr', type=float, default=0.1, help='Weight for expression regularization.')
    parser.add_argument('--reg_neighbor_radius', type=float, default=0.01, help="Radius for sampling neighbor points for regularization (in normalized space).")
    parser.add_argument('--no_pixel_variance', action='store_true', help='Disable per-pixel variance prediction.')
    parser.add_argument('--no_slice_variance', action='store_true', help='Disable per-slice variance prediction.')
    parser.add_argument('--n_levels_bias', type=int, default=0, help='Levels for bias network.')
    parser.add_argument('--weight_bias', type=float, default=0.1)
    parser.add_argument('--single_precision', action='store_true', help='Use single precision (fp32) instead of mixed precision.')
    parser.add_argument('--n_samples', type=int, default=4, help='Number of samples per point for PSF simulation.')
    parser.add_argument('--early_stopping_patience', type=int, default=5, help='Patience for early stopping.')
    parser.add_argument('--early_stopping_delta', type=float, default=1e-4, help='Min delta for early stopping.')
    parser.add_argument('--early_stopping_check_interval', type=int, default=200, help='Iteration interval for early stopping check.')
    parser.add_argument('--log_dir', type=str, default='runs', help='Directory for TensorBoard logs.')
    # --- Dropout预测相关参数 ---
    parser.add_argument('--no_dropout', action='store_true', 
                        help='Disable the dropout prediction network to handle zero-inflation.')
    parser.add_argument('--weight_dropout', type=float, default=2000.0, 
                        help='Weight for the dropout binary cross-entropy loss.')

    parsed_args = parser.parse_args()
    args_ns = argparse.Namespace(**vars(parsed_args))
    args_ns.dtype = torch.float32 if args_ns.single_precision else torch.float16
    return args_ns


def main():
    args = get_args()
    # data_basename = os.path.basename(args.input_data).split('.')[0]
    # subdir = data_basename.split('_')[0]
    # args.output_dir = f'/home/gongyuqiao/ur_annotation/NTF/mytrain/res/{subdir}/{data_basename}'
    # args.output_dir = f'/home/gongyuqiao/ur_annotation/NTF/mytrain/res/ABCA2/results/ABCA2_100wcell_200hvg'

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir,exist_ok=True)

    logging.info(f"Loading data from {args.input_data}")
    adata = anndata.read_h5ad(args.input_data)
    adata.obsm['spatial'] = adata.obsm['spatial'].astype(np.float32)

    indices = np.arange(adata.n_obs)
    train_indices, test_indices = train_test_split(indices, test_size=0.1, random_state=42)
    adata_train = adata[train_indices, :].copy()
    adata_test = adata[test_indices, :].copy()

    # adata_train = adata[adata.obs['group4']=='train', :].copy()
    # adata_test = adata[adata.obs['group4']=='test', :].copy()

    logging.info(f"Data split into training set ({adata_train.n_obs} cells) and test set ({adata_test.n_obs} cells).")

    raw_bbox_size = (adata_train.obsm['spatial'].max(0) - adata_train.obsm['spatial'].min(0)).max()
    spatial_scaling = 1.0 / raw_bbox_size if raw_bbox_size > 0 else 1.0
    logging.info(f"Scaling factor derived from training set: {spatial_scaling:.6f}")
    adata_train.obsm['spatial'] *= spatial_scaling

    t_start = time.time()
    trained_NTF = train(adata_train, args)
    trained_inr = trained_NTF.inr
    t_end = time.time()
    train_time_sec = t_end - t_start
    logging.info(f"Training finished. Time elapsed: {train_time_sec:.2f} seconds")

    logging.info("Evaluating on the test set...")
    raw_test_coords = adata_test.obsm['spatial']
    scaled_test_coords = raw_test_coords * spatial_scaling
    test_coords_tensor = torch.from_numpy(scaled_test_coords).float().to(args.device)

    t_start = time.time()
    prediction_results = sample_points(trained_inr, test_coords_tensor)
    t_end = time.time()
    infer_time_sec = t_end - t_start
    logging.info(f"Inference finished. Time elapsed: {infer_time_sec:.2f} seconds")

    predicted_expression = prediction_results["expression"].cpu().numpy()
    adata_test.layers['prediction'] = predicted_expression
    if args.no_dropout is False:
        adata_test.obsm['dropout_prob'] = prediction_results['dropout_prob'].cpu().numpy() if 'dropout_prob' in prediction_results else None

    true_expression = adata_test.X.toarray() if hasattr(adata_test.X, "toarray") else adata_test.X
    true_is_zero = (true_expression == 0.0)
    n_genes = adata_test.n_vars
    gene_names = adata_test.var_names if 'var_names' in dir(adata_test) and not adata_test.var_names.empty else [f'gene_{i}' for i in range(n_genes)]

    correlations = {}
    spearman_correlations = {}

    log_lines = []
    log_lines.append(f"Training time (seconds): {train_time_sec:.2f}\n")
    log_lines.append(f"Inference time (seconds): {infer_time_sec:.2f}\n")
    log_lines.append("--- Evaluation Results (Per-Gene Correlation on Non-Zero Expression) ---\n")

    logging.info("--- Evaluation ---")
    for i in range(n_genes):
        gene_name = gene_names[i]
        if 'dropout_prob' in prediction_results:
            prob = prediction_results['dropout_prob'].cpu().numpy()
            gt = true_is_zero[:, i]
            pred_p = prob[:, i]
            pred_lbl = (pred_p > 0.5)
            # 跳过全为0或全为1的情况
            if gt.sum() == 0 or gt.sum() == len(gt):
                print(f"{gene_names[i]}: skipped (all zero or all nonzero)")
                continue
            acc = accuracy_score(gt, pred_lbl)
            p, r, f1, _ = precision_recall_fscore_support(gt, pred_lbl, average='binary', zero_division=0)
            try:
                auc = roc_auc_score(gt, pred_p)
            except:
                auc = float('nan')
            log_lines.append(f"{gene_names[i]}: ACC={acc:.3f} Prec={p:.3f} Rec={r:.3f} F1={f1:.3f} AUC={auc:.3f}\n")

        true_vals_gene = true_expression[:, i]
        pred_vals_gene = predicted_expression[:, i]
        non_zero_mask_gene = true_vals_gene > 0
        if np.sum(non_zero_mask_gene) < 2:
            correlations[gene_name] = np.nan
            continue
        true_vals_filtered = true_vals_gene[non_zero_mask_gene]
        pred_vals_filtered = pred_vals_gene[non_zero_mask_gene]
        if np.var(true_vals_filtered) < 1e-10 or np.var(pred_vals_filtered) < 1e-10:
            correlations[gene_name] = np.nan
            continue
        try:
            # corr, p_value = pearsonr(true_vals_filtered, pred_vals_filtered)
            # sp_corr, sp_p_value = spearmanr(true_vals_filtered, pred_vals_filtered)
            corr, p_value = pearsonr(true_vals_gene, pred_vals_gene)
            sp_corr, sp_p_value = spearmanr(true_vals_gene, pred_vals_gene)
            correlations[gene_name] = corr
            spearman_correlations[gene_name] = sp_corr
            log_lines.append(f"Gene '{gene_name}': Pearson correlation = {corr:.4f} (p-value = {p_value:.2e}), Spearman correlation = {sp_corr:.4f} (p-value = {sp_p_value:.2e})\n")
        except ValueError:
            correlations[gene_name] = np.nan
            spearman_correlations[gene_name] = np.nan

    # save correlations to a csv file
    calculate_spatial_metrics(adata_test)
    adata_test.var['pearson_corr'] = [correlations.get(gene, np.nan) for gene in adata_test.var_names]
    adata_test.var['spearman_corr'] = [spearman_correlations.get(gene, np.nan) for gene in adata_test.var_names]
    adata_test.write_h5ad(f'{args.output_dir}/test_results.h5ad',compression='gzip')
    logging.info(f"Test results saved to {args.output_dir}/test_results.h5ad")
    adata_test.var.to_csv(f'{args.output_dir}/gene_metrics.csv')

    valid_correlations = [c for c in correlations.values() if not np.isnan(c)]
    valid_spearman = [c for c in spearman_correlations.values() if not np.isnan(c)]

    if valid_correlations:
        avg_corr = np.mean(valid_correlations)
        avg_sp_corr = np.mean(valid_spearman) if valid_spearman else float('nan')
        log_lines.append("--- Summary ---\n")
        log_lines.append(f"Average Pearson correlation across all valid genes: {avg_corr:.4f}\n")
        log_lines.append(f"Average Spearman correlation across all valid genes: {avg_sp_corr:.4f}\n")
    else:
        log_lines.append("No valid correlations could be computed.\n")

    if args.output_dir:
        with open(f'{args.output_dir}/res.log', 'w') as f:
            f.writelines(log_lines)
    else:
        for l in log_lines:
            print(l, end='')

if __name__ == '__main__':
    main()