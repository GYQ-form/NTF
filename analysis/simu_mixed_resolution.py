import argparse
import os
import logging
import numpy as np
import pandas as pd
import anndata
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scanpy as sc
from train_dropout_real import calculate_spatial_metrics
from NTF.sample import sample_points
from tqdm import tqdm

# Import our custom modules
from NTF.train import train 
from resolution_utils import create_mixed_resolution_dataset
from NTF.config import add_shared_args, process_args
from scipy.stats import pearsonr, spearmanr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def get_args():
    shared_parser = add_shared_args()
    parser = argparse.ArgumentParser(
        description="Run Mixed Resolution 3D Reconstruction Experiment",
        parents=[shared_parser] 
    )
    
    # Mixed Resolution Experiment Params
    exp_group = parser.add_argument_group('Experiment settings')
    exp_group.add_argument('--slice_interval_k', type=int, default=5, help='Take 1 slice every k slices')
    exp_group.add_argument('--high_res_intervals_m', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6, 10, 20, 30], 
                           help='List of m values to test. m=1 means all high res. m=20 means essentially only first is high res.')
    exp_group.add_argument('--bin_factors', type=float, nargs='+', default=[2.0,3.0,4.0,5.0,6.0,7.0,8.0],
                           help='List of bin factors (degradation levels) to test for low-res slices.')
    
    args = parser.parse_args()
    args = process_args(args)
    return args

def get_reconstruction(model, adata_full, scale_factor, device):

    rec_adata = adata_full.copy()
    coords = torch.from_numpy(rec_adata.obsm['spatial']).float().to(device)
    scaled_coords = coords * scale_factor
    
    model.eval()
    with torch.no_grad():
        res = sample_points(model, scaled_coords, batch_size=8192) 
        
    pred_expr = res['expression'].cpu().numpy()
    rec_adata.layers['prediction'] = pred_expr

    return rec_adata

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 1. read full data ---
    logging.info("Reading Ground Truth...")
    sim_adata = anndata.read_h5ad(args.input_data)

    # calculate spatial scaling factor
    raw_bbox_size = (sim_adata.obsm['spatial'].max(0) - sim_adata.obsm['spatial'].min(0)).max()
    spatial_scaling = 1.0 / raw_bbox_size if raw_bbox_size > 0 else 1.0
    logging.info(f"Scaling factor: {spatial_scaling:.6f}")

    results = []

    # --- 2. Experiment Loop ---
    # Loop over Bin Factors (How bad is the low-res?)
    for bf in args.bin_factors:
        # Loop over High-Res Frequency (How many high-res slices?)
        for m in args.high_res_intervals_m:
            
            exp_name = f"k{args.slice_interval_k}_m{m}_bf{bf}"
            logging.info(f"=== Starting Experiment: {exp_name} ===")
            logging.info(f"Configuration: Interval k={args.slice_interval_k}, "
                         f"High-Res every {m} slices, Low-Res Factor {bf}")
            
            # 2.1 Create Mixed Dataset
            train_adata = create_mixed_resolution_dataset(
                sim_adata,
                slice_interval_k=args.slice_interval_k,
                high_res_interval_m=m,
                bin_factor=bf
            )
            train_adata.obsm['spatial'] *= spatial_scaling
            n_cells = train_adata.shape[0]
            n_high = np.sum(train_adata.obs['resolution_type'] == 'high_res')
            n_low = np.sum(train_adata.obs['resolution_type'] == 'low_res')
            
            logging.info(f"Training Data: {n_cells} total points. "
                         f"High-Res: {n_high}, Low-Res: {n_low}")
            
            if n_cells < 100:
                logging.warning("Too few cells, skipping.")
                continue

            # 2.2 Train Model
            # NTF treats input as point cloud, so it handles mixed density naturally
            model_wrapper = train(train_adata, args)
            trained_inr = model_wrapper.inr
            
            # 2.3 Evaluate on FULL GT
            rec_adata =  get_reconstruction(trained_inr, sim_adata, spatial_scaling, args.device)
            
            # Calculate Metrics
            logging.info("Evaluating on Full Ground Truth...")
            pearson_corrs = []
            spearman_corrs = []

            for i in tqdm(range(rec_adata.n_vars), desc="Calculating gene correlations"):
                p = rec_adata.layers['prediction'][:, i]
                t = rec_adata.X[:, i]
                pearson_corrs.append(pearsonr(t, p)[0])
                spearman_corrs.append(spearmanr(t, p)[0])
                    
            # save correlations to a csv file
            calculate_spatial_metrics(rec_adata)
            rec_adata.var['pearson_corr'] = pearson_corrs
            rec_adata.var['spearman_corr'] = spearman_corrs
            rec_adata.write_h5ad(f'{args.output_dir}/rec_m{m}_bf{bf}.h5ad',compression='gzip')
            train_adata.write_h5ad(f'{args.output_dir}/train_m{m}_bf{bf}.h5ad',compression='gzip')
            rec_adata.var.to_csv(f'{args.output_dir}/m{m}_bf{bf}.csv')

            avg_corr = np.nanmean(pearson_corrs[:90])
            avg_sp_corr = np.nanmean(spearman_corrs[:90])
            avg_rmse = np.nanmean(rec_adata.var['rmse'][:90])
            avg_ssim3d = np.nanmean(rec_adata.var['ssim_3d'][:90])
            logging.info(f"Result {exp_name}: Avg Pearson = {avg_corr:.4f}")
            
            results.append({
                'bin_factor': bf,
                'high_res_m': m,
                'slice_interval_k': args.slice_interval_k,
                'n_train_points': n_cells,
                'ratio_high_res': n_high / n_cells,
                'avg_pearson': avg_corr,
                'avg_spearman': avg_sp_corr,
                'avg_mse': avg_rmse,
                'avg_ssim_3d': avg_ssim3d
            })
            
            # Clean up
            del model_wrapper, trained_inr, train_adata
            torch.cuda.empty_cache()
            
    # --- 3. Save & Plot ---
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.output_dir, 'mixed_res_results.csv'), index=False)
    
    # Pivot for Heatmap or Line Plot
    # X-axis: m (High-Res Interval), Y-axis: Pearson, Color/Line: Bin Factor
    try:
        import seaborn as sns
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='high_res_m', y='avg_pearson', hue='bin_factor', marker='o', palette='viridis')
        plt.title(f"Reconstruction Performance vs. Mixed Resolution (Slice Interval k={args.slice_interval_k})")
        plt.xlabel("High-Res Slice Interval (m)\n(1=All High, Higher=Fewer High)")
        plt.ylabel("Avg Pearson Correlation")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(args.output_dir, 'mixed_res_plot.png'))
    except Exception as e:
        logging.error(f"Plotting failed: {e}")

    logging.info("Experiment Finished.")

if __name__ == "__main__":
    main()