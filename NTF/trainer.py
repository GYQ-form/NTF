# ntf/trainer.py
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import pearsonr
import numpy as np
from tqdm.auto import tqdm # Use auto version for script/notebook compatibility
import os

from .model import NeuralTranscriptomicField
from .renderer import Renderer
from .losses import ReconstructionLoss, SmoothnessLoss

class Trainer:
    """
    Manages the training and evaluation loop for the NTF model,
    with support for tqdm progress bars and TensorBoard logging.
    """
    def __init__(self, model, dataset, config, device_idx = 0):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.renderer = Renderer(self.model, psf_std=config.get('psf_std', 0.1))
        self.recon_loss_fn = ReconstructionLoss()
        self.smooth_loss_fn = SmoothnessLoss(noise_std=config.get('smooth_noise_std', 0.05))
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.999)
        
        # --- TensorBoard Setup ---
        log_dir = config.get('log_dir', 'runs/ntf_experiment')
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logs will be saved to: {log_dir}")

        # --- Early stopping attributes ---
        self.early_stopping = config.get('early_stopping', False)
        self.patience = config.get('patience', 10)
        self.min_delta = config.get('min_delta', 0.001)
        self.best_metric = -float('inf')
        self.patience_counter = 0

    def _evaluate(self, dataloader, epoch):
        """Performs evaluation on a validation set."""
        self.model.eval()
        all_g_bar = []
        all_g_true = []
        
        val_progress = tqdm(dataloader, desc="Validating", leave=False)
        with torch.no_grad():
            for batch in val_progress:
                coords = batch["coords"].to(self.device)
                slice_ids = batch["slice_id"].to(self.device)
                g_true = batch["genes"].to(self.device)

                g_bar, _ = self.renderer.render_spots(coords, slice_ids, K=self.config.get('K_samples', 16))
                
                all_g_bar.append(g_bar.cpu().numpy())
                all_g_true.append(g_true.cpu().numpy())

        all_g_bar = np.concatenate(all_g_bar, axis=0)
        all_g_true = np.concatenate(all_g_true, axis=0)
        
        corrs = []
        for i in range(all_g_true.shape[1]):
            # Filter out genes with zero variance to avoid pearsonr warnings
            if np.std(all_g_true[:, i]) > 1e-6 and np.std(all_g_bar[:, i]) > 1e-6:
                corr, _ = pearsonr(all_g_true[:, i], all_g_bar[:, i])
                if not np.isnan(corr):
                    corrs.append(corr)
        
        mean_corr = np.mean(corrs) if corrs else 0.0
        self.writer.add_scalar('Validation/PearsonCorr', mean_corr, epoch)
        return mean_corr

    def train(self):
        """Main training loop."""
        val_loader = None
        if self.early_stopping:
            val_size = int(0.1 * len(self.dataset))
            train_size = len(self.dataset) - val_size
            train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size])
            val_loader = DataLoader(val_dataset, batch_size=self.config.get('batch_size',4096))
        else:
            train_dataset = self.dataset
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.get('batch_size',4096), shuffle=True, num_workers=4, pin_memory=True)

        epoch_pbar = tqdm(range(self.config.get('epochs',200)), desc="Training Progress")

        try:
            for epoch in epoch_pbar:
                self.model.train()
                total_loss = 0
                
                batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.get('epochs',200)}", leave=False)

                for batch in batch_pbar:
                    coords = batch["coords"].to(self.device)
                    slice_ids = batch["slice_id"].to(self.device)
                    g_true = batch["genes"].to(self.device)

                    self.optimizer.zero_grad(set_to_none=True)
                    
                    g_bar, g_var = self.renderer.render_spots(coords, slice_ids, K=self.config.get('K_samples',16))
                    recon_loss = self.recon_loss_fn(g_true, g_bar, g_var)
                    
                    loss_v, loss_g = self.smooth_loss_fn(self.model, coords, slice_ids)
                    
                    total_loss_val = (recon_loss + 
                                    self.config.get('lambda_v',0.1) * loss_v +
                                    self.config.get('lambda_g',0.2) * loss_g)
                    
                    total_loss_val.backward()
                    self.optimizer.step()
                    total_loss += total_loss_val.item()
                    
                    batch_pbar.set_postfix({'loss': f'{total_loss_val.item():.4f}'})

                self.scheduler.step()
                avg_loss = total_loss / len(train_loader)
                
                # --- Logging to TensorBoard ---
                self.writer.add_scalar('Loss/train', avg_loss, epoch)
                self.writer.add_scalar('LearningRate', self.scheduler.get_last_lr()[0], epoch)

                # Update outer progress bar description
                postfix_metrics = {'avg_loss': f'{avg_loss:.4f}'}

                # --- Early stopping check ---
                if self.early_stopping and val_loader:
                    val_metric = self._evaluate(val_loader, epoch)
                    postfix_metrics['val_corr'] = f'{val_metric:.4f}'
                    
                    if val_metric > self.best_metric + self.min_delta:
                        self.best_metric = val_metric
                        self.patience_counter = 0
                        # Create checkpoint directory if it doesn't exist
                        ckpt_path = self.config.get('checkpoint_path', 'best_model.pth')
                        ckpt_dir = os.path.dirname(ckpt_path)
                        if ckpt_dir: os.makedirs(ckpt_dir, exist_ok=True)
                        torch.save(self.model.state_dict(), ckpt_path)
                    else:
                        self.patience_counter += 1
                    
                    if self.patience_counter >= self.patience:
                        print(f"Early stopping triggered after {self.patience} epochs with no improvement.")
                        break
                    
                epoch_pbar.set_postfix(postfix_metrics)

        finally:
            self.writer.close()
            print("Training finished. TensorBoard writer closed.")