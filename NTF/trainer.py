# # ntf_torch/trainer.py
# import torch
# from tqdm import tqdm
# from datetime import datetime
# from torch.utils.tensorboard import SummaryWriter

# from .renderer import volume_render
# from .losses import reconstruction_loss, geometric_loss, smoothness_loss

# class Trainer:
#     def __init__(self, model, dataset, config, device, log_dir=None):
#         self.model = model
#         self.dataset = dataset
#         self.config = config
#         self.device = device
        
#         self.optimizer = torch.optim.Adam(
#             self.model.parameters(), 
#             lr=self.config['training']['lr']
#         )
        
#         self.dataloader = torch.utils.data.DataLoader(
#             self.dataset,
#             batch_size=self.config['training']['batch_size'],
#             shuffle=True,
#             num_workers=4
#         )
        
#         self.bounds = (
#             self.dataset.bounds[0].to(self.device),
#             self.dataset.bounds[1].to(self.device)
#         )

#         # TensorBoard support
#         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#         self.log_dir = f"runs/ntf_experiment_{timestamp}" if log_dir is None else f"{log_dir}/runs/ntf_experiment_{timestamp}"
#         self.writer = SummaryWriter(self.log_dir)

#     def train(self):
#         global_step = 0
#         for epoch in range(self.config['training']['epochs']):
#             self.model.train()
#             pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.config['training']['epochs']}")
            
#             for batch in pbar:
#                 origins = batch['origin'].to(self.device)
#                 target_genes = batch['target_genes'].to(self.device)
                
#                 self.optimizer.zero_grad()
                
#                 rendered_genes = volume_render(
#                     self.model,
#                     origins,
#                     self.config['data']['slice_thickness'],
#                     self.config['rendering']['num_samples']
#                 )
                
#                 l_recon = reconstruction_loss(rendered_genes, target_genes)
#                 l_geom = geometric_loss(self.model, self.bounds)
#                 l_smooth = smoothness_loss(self.model, self.bounds)
                
#                 total_loss = (
#                     self.config['losses']['w_recon'] * l_recon +
#                     self.config['losses']['w_geom'] * l_geom +
#                     self.config['losses']['w_smooth'] * l_smooth
#                 )
                
#                 total_loss.backward()
#                 self.optimizer.step()

#                 # record losses to TensorBoard
#                 self.writer.add_scalar('Loss/total', total_loss.item(), global_step)
#                 self.writer.add_scalar('Loss/reconstruction', l_recon.item(), global_step)
#                 self.writer.add_scalar('Loss/geometric', l_geom.item(), global_step)
#                 self.writer.add_scalar('Loss/smoothness', l_smooth.item(), global_step)
#                 global_step += 1
                
#                 pbar.set_postfix(
#                     loss=f"{total_loss.item():.6f}",
#                     recon=f"{l_recon.item():.6f}"
#                 )

#             # record learning rate to TensorBoard
#             self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], epoch)

#             if (epoch + 1) % self.config['training']['save_every'] == 0:
#                 save_path = f"model_epoch_{epoch+1}.pth"
#                 torch.save(self.model.state_dict(), save_path)
#                 print(f"Model saved to {save_path}")

#         # turn off writer
#         self.writer.close()

import torch
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from torch.utils.data import DataLoader, random_split

from .renderer import volume_render
from .losses import reconstruction_loss, geometric_loss, smoothness_loss
from scipy.stats import pearsonr

class Trainer:
    def __init__(self, model, train_dataset, config, device, log_dir=None):
        self.model = model
        self.dataset = train_dataset
        self.config = config
        self.device = device
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config['training']['lr']
        )
        
        self.bounds = (
            self.dataset.bounds[0].to(self.device),
            self.dataset.bounds[1].to(self.device)
        )

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = f"runs/ntf_experiment_{timestamp}" if log_dir is None else f"{log_dir}/runs/ntf_experiment_{timestamp}"
        self.writer = SummaryWriter(self.log_dir)

        # --- for Early Stopping ---
        self.use_early_stopping = self.config['training']['early_stopping'].get('enabled', False)
        
        if self.use_early_stopping:
            val_fraction = self.config['training']['early_stopping'].get('val_fraction', 0.1)
            if not (0 < val_fraction < 1):
                raise ValueError("val_fraction must be between 0 and 1.")
            
            # 使用PyTorch的random_split进行高效划分
            dataset_size = len(self.dataset)
            val_size = int(val_fraction * dataset_size)
            train_size = dataset_size - val_size
            
            print(f"Internal split: {train_size} for training, {val_size} for validation.")
            
            # 设置一个固定的生成器以确保可重复性
            generator = torch.Generator().manual_seed(42)
            train_subset, val_subset = random_split(self.dataset, [train_size, val_size], generator=generator)
            
            self.train_dataloader = DataLoader(train_subset, batch_size=self.config['training']['batch_size'], shuffle=True, num_workers=4)
            self.val_dataloader = DataLoader(val_subset, batch_size=self.config['training']['batch_size'], shuffle=False, num_workers=4)
            
            self.patience = self.config['training']['early_stopping']['patience']
            self.min_delta = self.config['training']['early_stopping']['min_delta']
            self.patience_counter = 0
            self.best_val_metric = -np.inf
        else:
            print("Early stopping is disabled. Using the entire dataset for training.")
            self.train_dataloader = DataLoader(self.dataset, batch_size=self.config['training']['batch_size'], shuffle=True, num_workers=4)
            self.val_dataloader = None # 没有验证集
        # ------------------------------------

    def _validate(self) -> float:
        """
        在一个epoch后执行验证，并计算平均皮尔逊相关系数作为评估指标。
        """
        self.model.eval()
        all_reconstructed = []
        all_ground_truth = []
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                origins = batch['origin'].to(self.device)
                target_genes = batch['target_genes'] # Keep on CPU for now
                
                reconstructed_genes = volume_render(
                    self.model, origins,
                    self.config['data']['slice_thickness'],
                    self.config['rendering']['num_samples']
                )
                all_reconstructed.append(reconstructed_genes.cpu())
                all_ground_truth.append(target_genes)
        
        # 拼接所有批次的结果
        reconstructed_np = torch.cat(all_reconstructed).numpy()
        ground_truth_np = torch.cat(all_ground_truth).numpy()
        
        # 逐基因计算相关系数
        correlations = []
        for i in range(ground_truth_np.shape[1]):
            corr, _ = pearsonr(ground_truth_np[:, i], reconstructed_np[:, i])
            if not np.isnan(corr):
                correlations.append(corr)
        
        # 返回忽略NaN值的平均相关系数
        return np.mean(correlations) if correlations else 0.0

    def train(self) -> str:
        """执行完整的训练流程，包含早停逻辑。"""

        save_dir = self.config['training'].get('save_dir', '.')
        final_model_path = os.path.join(save_dir, "final_model.pth")
        best_model_path = os.path.join(save_dir, "best_model.pth")

        global_step = 0
        pbar = tqdm(range(self.config['training']['epochs']), desc="Training Progress")
        for epoch in pbar:
            self.model.train()
            total_train_loss = 0
            
            for batch in self.train_dataloader:
                origins = batch['origin'].to(self.device)
                target_genes = batch['target_genes'].to(self.device)
                
                self.optimizer.zero_grad()
                
                rendered_genes = volume_render(
                    self.model,
                    origins,
                    self.config['data']['slice_thickness'],
                    self.config['rendering']['num_samples']
                )
                
                l_recon = reconstruction_loss(rendered_genes, target_genes,
                                              positive_weight=self.config['losses'].get('positive_weight', 10.0))
                l_geom = geometric_loss(self.model, self.bounds)
                l_smooth = smoothness_loss(self.model, self.bounds)
                
                total_loss = (
                    self.config['losses']['w_recon'] * l_recon +
                    self.config['losses']['w_geom'] * l_geom +
                    self.config['losses']['w_smooth'] * l_smooth
                )
                
                total_loss.backward()
                self.optimizer.step()

                # record losses to TensorBoard
                self.writer.add_scalar('Loss/total', total_loss.item(), global_step)
                self.writer.add_scalar('Loss/reconstruction', l_recon.item(), global_step)
                self.writer.add_scalar('Loss/geometric', l_geom.item(), global_step)
                self.writer.add_scalar('Loss/smoothness', l_smooth.item(), global_step)
                global_step += 1
                total_train_loss += total_loss.item()
            
            avg_train_loss = total_train_loss / len(self.train_dataloader)
            self.writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
            self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # --- 验证和早停检查 ---
            if self.val_dataloader:
                current_val_corr = self._validate()
                self.writer.add_scalar('Metric/validation_pearson_corr', current_val_corr, epoch)
                
                # 更新epoch级别进度条的后缀信息
                pbar.set_postfix(
                    train_loss=f"{avg_train_loss:.4f}", 
                    val_corr=f"{current_val_corr:.4f}",
                    best_corr=f"{self.best_val_metric:.4f}"
                )

                if self.use_early_stopping:
                    # 现在是相关系数越高越好
                    if current_val_corr - self.best_val_metric > self.min_delta:
                        self.best_val_metric = current_val_corr
                        self.patience_counter = 0
                        torch.save(self.model.state_dict(), best_model_path)
                    else:
                        self.patience_counter += 1
                    
                    if self.patience_counter >= self.patience:
                        print("Early stopping triggered.")
                        self.writer.close()
                        return best_model_path
            else:
                pbar.set_postfix(train_loss=f"{avg_train_loss:.4f}")
            # --------------------------

        # 训练正常结束 (未触发早停或早停被关闭)
        print("Training finished after all epochs.")
        torch.save(self.model.state_dict(), final_model_path)
        self.writer.close()
        
        if self.use_early_stopping:
            print(f"Returning best model saved at {best_model_path}")
            return best_model_path
        else:
            print(f"Early stopping disabled. Returning final model saved at {final_model_path}")
            return final_model_path