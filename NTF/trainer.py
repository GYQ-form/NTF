# trainer.py
import torch
from tqdm import tqdm
from renderer import volume_render
from losses import reconstruction_loss, geometric_loss, smoothness_loss

def train(config):
    # 1. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 加载数据
    from dataset import SpatialOmicsDataset
    dataset = SpatialOmicsDataset(config['data']['path'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)
    
    # 3. 初始化模型
    from model import NeuralTranscriptomicField
    model = NeuralTranscriptomicField(
        num_genes=dataset.expressions.shape[1],
        config=config['model']
    ).to(device)

    # 4. 初始化优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])
    
    # 5. 训练循环
    for epoch in range(config['training']['epochs']):
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            origins = batch['origin'].to(device)
            target_genes = batch['target_genes'].to(device)
            
            optimizer.zero_grad()
            
            # 渲染基因表达
            rendered_genes = volume_render(
                model,
                origins,
                config['data']['slice_thickness'],
                config['rendering']['num_samples']
            )
            
            # 计算各项损失
            l_recon = reconstruction_loss(rendered_genes, target_genes)
            l_geom = geometric_loss(model, (dataset.bounds[0].to(device), dataset.bounds[1].to(device)))
            l_smooth = smoothness_loss(model, (dataset.bounds[0].to(device), dataset.bounds[1].to(device)))
            
            # 加权总损失
            total_loss = (
                config['losses']['w_recon'] * l_recon +
                config['losses']['w_geom'] * l_geom +
                config['losses']['w_smooth'] * l_smooth
            )
            
            # 反向传播和优化
            total_loss.backward()
            optimizer.step()
            
            tqdm.write(f"Loss: {total_loss.item():.6f} "
                       f"(Recon: {l_recon.item():.6f}, Geom: {l_geom.item():.6f}, Smooth: {l_smooth.item():.6f})")

        # 保存模型
        if (epoch + 1) % config['training']['save_every'] == 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")

# main.py
import yaml
from trainer import train

if __name__ == "__main__":
    with open("configs/default_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    train(config)