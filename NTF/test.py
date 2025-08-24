# ntf_torch/trainer.py
import torch
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split, Subset
import os
import numpy as np
from scipy.stats import pearsonr

from .renderer import volume_render
from .losses import reconstruction_loss, density_smoothness_loss, gene_smoothness_loss

class Trainer:
    def __init__(self, model, dataset, config, device):
        self.model = model
        self.dataset = dataset
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
        self.log_dir = f"runs/ntf_experiment_{timestamp}"
        self.writer = SummaryWriter(self.log_dir)

        # --- START: 内部划分验证集 ---
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
        # --- END: 内部划分验证集 ---

    def _validate(self) -> float:
        # ... (此函数无需修改) ...

    def train(self) -> str:
        # ... (此函数大部分逻辑不变，只需确保在调用_validate前检查self.val_dataloader即可) ...
        # ...
            # --- 在调用验证前增加检查 ---
            if self.val_dataloader:
                current_val_metric = self._validate()
                self.writer.add_scalar('Metric/validation_pearson_corr', current_val_metric, epoch)
                
                pbar_postfix = {"train_loss": f"{avg_train_loss:.4f}", "val_corr": f"{current_val_metric:.4f}"}
                
                if self.use_early_stopping:
                    pbar_postfix["best_corr"] = f"{self.best_val_metric:.4f}"
                    if current_val_metric - self.best_val_metric > self.min_delta:
                        self.best_val_metric = current_val_metric
                        self.patience_counter = 0
                        print(f"Validation correlation improved to {current_val_metric:.4f}. Saving best model...")
                        torch.save(self.model.state_dict(), best_model_path)
                    else:
                        self.patience_counter += 1
                    
                    if self.patience_counter >= self.patience:
                        print("Early stopping triggered.")
                        self.writer.close()
                        return best_model_path
                pbar.set_postfix(pbar_postfix)
            else:
                 pbar.set_postfix(train_loss=f"{avg_train_loss:.4f}")

        # ... (函数结尾部分不变) ...