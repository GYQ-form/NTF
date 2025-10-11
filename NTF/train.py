from argparse import Namespace
import time
import datetime
import logging
import os
import torch
import torch.optim as optim
import anndata
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from .models import NeuralTranscriptomicField, GeneINR, D_LOSS, S_LOSS, DS_LOSS, DO_LOSS, E_REG, B_REG
from .data import SpatialOmicsDataset
from .utils import MovingAverage

def train(
    adata: anndata.AnnData,
    args: Namespace
) -> GeneINR:
    
    # --- TensorBoard and Logging Setup ---
    log_dir_base = getattr(args, 'log_dir', 'runs')
    run_name = 'NTF_' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join(log_dir_base, run_name)
    writer = SummaryWriter(log_dir)
    logging.info(f"TensorBoard logs will be saved to: {log_dir}")

    # --- Early Stopping Parameters ---
    patience = getattr(args, 'early_stopping_patience', 5)
    min_delta = getattr(args, 'early_stopping_delta', 1e-4)
    check_interval = getattr(args, 'early_stopping_check_interval', 500)
    
    # Create training dataset
    dataset = SpatialOmicsDataset(adata, args.slice_id)
    if args.n_epochs is not None:
        args.n_iter = args.n_epochs * (dataset.v.shape[0] // args.batch_size)

    bounding_box = dataset.bounding_box
    
    model = NeuralTranscriptomicField(
        n_genes=dataset.n_genes, n_slices=dataset.n_slices,
        resolution=dataset.resolution,
        bounding_box=bounding_box, args=args,
        pos_weight=dataset.nonzero_to_zero_ratio,
    )
    
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.99),
        eps=1e-15, weight_decay=1e-2
    )
    
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=[int(m * args.n_iter) for m in args.milestones],
        gamma=args.gamma,
    )
    
    fp16 = not args.single_precision
    scaler = torch.amp.GradScaler('cuda',enabled=fp16)
    
    model.train()
    loss_weights = {
        D_LOSS: 1, S_LOSS: 1, B_REG: args.weight_bias, E_REG: args.weight_expr, DO_LOSS: args.weight_dropout,
    }
    average = MovingAverage(1 - 0.001)
    
    # --- Early Stopping Initialization ---
    patience_counter = 0
    best_loss = float('inf')
    warmup_iters = 2 * check_interval
    
    logging.info("NeuralTranscriptomicField training starts.")
    
    # --- tqdm Progress Bar ---
    # 使用tqdm包装主循环，并设置一个描述
    pbar = tqdm(range(1, args.n_iter + 1), desc="Training NeuralTranscriptomicField")

    for i in pbar:
        batch = dataset.get_batch(args.batch_size, args.device)
        
        with torch.amp.autocast('cuda',enabled=fp16):
            losses = model(**batch)
            loss = losses[DS_LOSS] + \
                loss_weights.get(DO_LOSS, 0) * losses.get(DO_LOSS, 0) + \
                loss_weights.get(B_REG, 0) * losses.get(B_REG, 0) + \
                loss_weights.get(E_REG, 0) * losses.get(E_REG, 0)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        # Update moving average for all losses
        for k in losses:
            if losses[k] is not None:
                average(k, losses[k].item())
        average('All', loss.item())
        
        # --- Update tqdm progress bar with live metrics ---
        postfix_dict = {
            'DSLoss': f'{average[DS_LOSS]:.4f}',
            'MSE': f'{average[D_LOSS]:.4f}',
            'LR': f'{optimizer.param_groups[0]["lr"]:.1e}'
        }
        if not args.no_dropout:
            postfix_dict['DOLoss'] = f'{average[DO_LOSS]:.4f}'
        pbar.set_postfix(postfix_dict)

        # --- Periodic Logging and Early Stopping Check ---
        if i % check_interval == 0 or i == args.n_iter:
            current_loss = average['All']
            
            # --- Log to TensorBoard ---
            writer.add_scalar('Loss/total_moving_avg', current_loss, i)
            # writer.add_scalar('Loss/mse_moving_avg', average[D_LOSS], i)
            # writer.add_scalar('Loss/logvar_moving_avg', average[S_LOSS], i)
            # if E_REG in average:
            #     writer.add_scalar('Loss/reg_image_moving_avg', average[E_REG], i)
            # if B_REG in average:
            #     writer.add_scalar('Loss/reg_bias_moving_avg', average[B_REG], i)
            # writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], i)

            # --- Early Stopping Logic ---
            if i > warmup_iters:
                if best_loss - current_loss > min_delta:
                    best_loss = current_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Log patience to console and TensorBoard
                tqdm.write(f"Iter: {i}, Loss: {current_loss:.6f}, Best: {best_loss:.6f}, Patience: {patience_counter}/{patience}")
                writer.add_scalar('EarlyStopping/patience_counter', patience_counter, i)

                if patience_counter >= patience:
                    tqdm.write(f"Early stopping triggered at iteration {i}.")
                    break
        
        # Scheduler Step
        scheduler.step()

    pbar.close() # 关闭进度条
    writer.close() # 关闭TensorBoard writer
    logging.info("Training finished.")
    
    return model