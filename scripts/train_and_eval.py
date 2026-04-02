import torch
import anndata
import numpy as np
import logging
import argparse
import os
import time
import matplotlib
matplotlib.use('Agg')  # Support headless server rendering
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
from NTF.train import train
from NTF.sample import sample_points
from NTF.config import add_shared_args, process_args
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from NTF.helper_func import calculate_spatial_metrics

def get_args():
    shared_parser = add_shared_args(description="Train and evaluate NeuroTField model.")
    parser = argparse.ArgumentParser(
        description="Train and evaluate NTF model.",
        parents=[shared_parser]
    )

    split_group = parser.add_argument_group('Train/Test Split')
    split_group.add_argument(
        '--split_mode', type=str, default='random', choices=['random', 'label'],
        help=(
            'How to split the data into train/test sets. '
            '"random": randomly split by --test_size fraction (default). '
            '"label": use a column in adata.obs to determine the split (requires --split_column, '
            '--train_label, --test_label).'
        )
    )
    split_group.add_argument(
        '--test_size', type=float, default=0.1,
        help='Fraction of data to use for the test set when --split_mode=random. Default: 0.1.'
    )
    split_group.add_argument(
        '--split_column', type=str, default=None,
        help='Column in adata.obs used for label-based splitting (required when --split_mode=label).'
    )
    split_group.add_argument(
        '--train_label', type=str, default=None,
        help='Value in --split_column that identifies training cells (required when --split_mode=label).'
    )
    split_group.add_argument(
        '--test_label', type=str, default=None,
        help='Value in --split_column that identifies test cells (required when --split_mode=label).'
    )

    args = parser.parse_args()
    args = process_args(args)

    # Validate label-based split arguments
    if args.split_mode == 'label':
        missing = [f for f, v in [
            ('--split_column', args.split_column),
            ('--train_label', args.train_label),
            ('--test_label', args.test_label),
        ] if v is None]
        if missing:
            parser.error(f"--split_mode=label requires: {', '.join(missing)}")

    return args


def main():
    args = get_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logging.info(f"Loading data from {args.input_data}")
    adata = anndata.read_h5ad(args.input_data)
    adata.obsm['spatial'] = adata.obsm['spatial'].astype(np.float32)

    # --- Train/Test Split ---
    if args.split_mode == 'random':
        indices = np.arange(adata.n_obs)
        train_indices, test_indices = train_test_split(indices, test_size=args.test_size, random_state=42)
        adata_train = adata[train_indices, :].copy()
        adata_test = adata[test_indices, :].copy()
    else:  # label-based split
        col = args.split_column
        if col not in adata.obs.columns:
            raise ValueError(f"Column '{col}' not found in adata.obs. Available columns: {list(adata.obs.columns)}")
        train_mask = adata.obs[col] == args.train_label
        test_mask = adata.obs[col] == args.test_label
        if train_mask.sum() == 0:
            raise ValueError(f"No cells found with {col}='{args.train_label}'.")
        if test_mask.sum() == 0:
            raise ValueError(f"No cells found with {col}='{args.test_label}'.")
        adata_train = adata[train_mask, :].copy()
        adata_test = adata[test_mask, :].copy()

    logging.info(f"Data split into training set ({adata_train.n_obs} cells) and test set ({adata_test.n_obs} cells).")

    raw_bbox_size = (adata_train.obsm['spatial'].max(0) - adata_train.obsm['spatial'].min(0)).max()
    spatial_scaling = 1.0 / raw_bbox_size if raw_bbox_size > 0 else 1.0
    logging.info(f"Scaling factor derived from training set: {spatial_scaling:.6f}")
    adata_train.obsm['spatial'] *= spatial_scaling

    # train() automatically logs training time
    trained_NTF = train(adata_train, args)
    trained_inr = trained_NTF.inr

    logging.info("Evaluating on the test set...")
    raw_test_coords = adata_test.obsm['spatial']
    scaled_test_coords = raw_test_coords * spatial_scaling
    test_coords_tensor = torch.from_numpy(scaled_test_coords).float().to(args.device)

    t_start = time.time()
    prediction_results = sample_points(trained_inr, test_coords_tensor)
    infer_time_sec = time.time() - t_start
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
            # Skip genes where all values are zero or all nonzero
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
            corr, p_value = pearsonr(true_vals_gene, pred_vals_gene)
            sp_corr, sp_p_value = spearmanr(true_vals_gene, pred_vals_gene)
            correlations[gene_name] = corr
            spearman_correlations[gene_name] = sp_corr
            log_lines.append(f"Gene '{gene_name}': Pearson correlation = {corr:.4f} (p-value = {p_value:.2e}), Spearman correlation = {sp_corr:.4f} (p-value = {sp_p_value:.2e})\n")
        except ValueError:
            correlations[gene_name] = np.nan
            spearman_correlations[gene_name] = np.nan

    # Save correlations and metrics
    calculate_spatial_metrics(adata_test)
    adata_test.var['pearson_corr'] = [correlations.get(gene, np.nan) for gene in adata_test.var_names]
    adata_test.var['spearman_corr'] = [spearman_correlations.get(gene, np.nan) for gene in adata_test.var_names]
    adata_test.write_h5ad(f'{args.output_dir}/test_results.h5ad', compression='gzip')
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
