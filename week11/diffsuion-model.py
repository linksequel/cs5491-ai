"""
Diffusion Model Demo - DDPM实现
在MNIST数据集上训练一个简单的扩散模型

参考论文: Denoising Diffusion Probabilistic Models (Ho et al., 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


# ==================== 超参数设置 ====================
class Config:
    # 数据集参数
    image_size = 28
    channels = 1
    batch_size = 128
    
    # Diffusion参数
    timesteps = 1000  # 扩散步数
    beta_start = 0.0001  # β起始值
    beta_end = 0.02  # β结束值
    
    # 训练参数
    epochs = 20
    lr = 2e-4
    device = torch.device("mps" if torch.backends.mps.is_available() 
                         else "cuda" if torch.cuda.is_available() 
                         else "cpu")
    
    # 保存路径
    save_dir = "diffusion_results"
    model_path = "diffusion_model.pth"


# ==================== Diffusion工具函数 ====================
def linear_beta_schedule(timesteps, beta_start, beta_end):
    """线性β调度"""
    return torch.linspace(beta_start, beta_end, timesteps)


def prepare_noise_schedule(timesteps, beta_start, beta_end):
    """
    准备噪声调度表
    预计算所有需要的系数以加速训练
    """
    # β值
    betas = linear_beta_schedule(timesteps, beta_start, beta_end)
    
    # α = 1 - β
    alphas = 1.0 - betas
    
    # α̅ = ∏(1-βᵢ) 累积乘积
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    
    # 计算后验分布q(x_{t-1}|x_t,x_0)所需的系数
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    
    # 后验方差
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    
    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'alphas_cumprod_prev': alphas_cumprod_prev,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
        'sqrt_recip_alphas': sqrt_recip_alphas,
        'posterior_variance': posterior_variance,
    }


# ==================== UNet模型架构 ====================
class SinusoidalPositionEmbeddings(nn.Module):
    """时间步的正弦位置编码"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class Block(nn.Module):
    """基础卷积块"""
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        # 第一个卷积
        h = self.bnorm1(self.relu(self.conv1(x)))
        # 时间嵌入
        time_emb = self.relu(self.time_mlp(t))
        # 扩展时间嵌入维度并加到特征上
        time_emb = time_emb[(..., ) + (None, ) * 2]
        h = h + time_emb
        # 第二个卷积
        h = self.bnorm2(self.relu(self.conv2(h)))
        # 下采样或上采样
        return self.transform(h)


class SimpleUNet(nn.Module):
    """
    简化的UNet架构用于噪声预测
    输入: 噪声图像 x_t 和时间步 t
    输出: 预测的噪声 ε
    """
    def __init__(self, image_channels=1, time_emb_dim=32):
        super().__init__()
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # 初始投影
        self.conv0 = nn.Conv2d(image_channels, 64, 3, padding=1)
        
        # 下采样
        self.downs = nn.ModuleList([
            Block(64, 128, time_emb_dim),
            Block(128, 256, time_emb_dim),
        ])
        
        # 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
        )
        
        # 上采样
        self.ups = nn.ModuleList([
            Block(256, 128, time_emb_dim, up=True),
            Block(128, 64, time_emb_dim, up=True),
        ])
        
        # 输出层
        self.output = nn.Conv2d(64, image_channels, 1)

    def forward(self, x, timestep):
        # 时间嵌入
        t = self.time_mlp(timestep)
        
        # 初始卷积
        x = self.conv0(x)
        
        # UNet下采样路径
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        
        # 瓶颈层
        x = self.bottleneck(x)
        
        # UNet上采样路径（带跳跃连接）
        for up, residual in zip(self.ups, reversed(residual_inputs)):
            x = torch.cat([x, residual], dim=1)
            x = up(x, t)
        
        return self.output(x)


# ==================== Diffusion模型 ====================
class DiffusionModel:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        # 准备噪声调度
        self.noise_schedule = prepare_noise_schedule(
            config.timesteps, 
            config.beta_start, 
            config.beta_end
        )
        
        # 将调度参数移到设备上
        for k, v in self.noise_schedule.items():
            self.noise_schedule[k] = v.to(self.device)
        
        # 初始化模型
        self.model = SimpleUNet(image_channels=config.channels).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        
        print(f"使用设备: {self.device}")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def q_sample(self, x_0, t, noise=None):
        """
        前向扩散过程: q(x_t|x_0)
        根据公式: x_t = √α̅_t * x_0 + √(1-α̅_t) * ε
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alphas_cumprod_t = self.noise_schedule['sqrt_alphas_cumprod'][t]
        sqrt_one_minus_alphas_cumprod_t = self.noise_schedule['sqrt_one_minus_alphas_cumprod'][t]
        
        # 调整维度以匹配batch
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[:, None, None, None]
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise
    
    def p_losses(self, x_0, t):
        """
        计算训练损失
        预测噪声并与真实噪声计算MSE
        """
        noise = torch.randn_like(x_0)
        x_noisy, _ = self.q_sample(x_0, t, noise)
        predicted_noise = self.model(x_noisy, t)
        loss = F.mse_loss(predicted_noise, noise)
        return loss
    
    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        """
        反向去噪单步: p(x_{t-1}|x_t)
        """
        betas_t = self.noise_schedule['betas'][t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.noise_schedule['sqrt_one_minus_alphas_cumprod'][t][:, None, None, None]
        sqrt_recip_alphas_t = self.noise_schedule['sqrt_recip_alphas'][t][:, None, None, None]
        
        # 预测噪声
        predicted_noise = self.model(x, t)
        
        # 计算均值
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.noise_schedule['posterior_variance'][t][:, None, None, None]
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def sample(self, batch_size=16):
        """
        从纯噪声开始采样生成图像
        """
        self.model.eval()
        
        # 从纯噪声开始
        img = torch.randn(
            (batch_size, self.config.channels, self.config.image_size, self.config.image_size),
            device=self.device
        )
        
        imgs = []
        
        # 逐步去噪
        for i in tqdm(reversed(range(0, self.config.timesteps)), desc='采样中', total=self.config.timesteps):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            img = self.p_sample(img, t, i)
            
            # 保存某些中间步骤
            if i % 200 == 0:
                imgs.append(img.cpu())
        
        return img, imgs
    
    def train(self, dataloader):
        """训练循环"""
        self.model.train()
        
        for epoch in range(self.config.epochs):
            pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{self.config.epochs}')
            total_loss = 0
            
            for batch_idx, (images, _) in enumerate(pbar):
                images = images.to(self.device)
                batch_size = images.shape[0]
                
                # 随机采样时间步
                t = torch.randint(0, self.config.timesteps, (batch_size,), device=self.device).long()
                
                # 计算损失
                loss = self.p_losses(images, t)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / len(dataloader)
            print(f'Epoch {epoch+1}, 平均损失: {avg_loss:.4f}')
            
            # 每个epoch结束后采样一些图像
            if (epoch + 1) % 5 == 0:
                self.visualize_samples(epoch + 1)
    
    def visualize_samples(self, epoch):
        """可视化生成的样本"""
        os.makedirs(self.config.save_dir, exist_ok=True)
        
        samples, intermediate = self.sample(batch_size=16)
        
        # 保存最终样本
        samples = (samples + 1) / 2  # 从[-1,1]转到[0,1]
        save_image(samples, f'{self.config.save_dir}/samples_epoch_{epoch}.png', nrow=4)
        
        # 保存中间步骤
        if intermediate:
            all_imgs = torch.cat(intermediate, dim=0)
            all_imgs = (all_imgs + 1) / 2
            save_image(all_imgs, f'{self.config.save_dir}/process_epoch_{epoch}.png', nrow=16)
        
        print(f'样本已保存到 {self.config.save_dir}/')
    
    def save_model(self):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.config.model_path)
        print(f'模型已保存到 {self.config.model_path}')
    
    def load_model(self):
        """加载模型"""
        checkpoint = torch.load(self.config.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f'模型已从 {self.config.model_path} 加载')


# ==================== 数据加载 ====================
def get_dataloader(config):
    """准备MNIST数据加载器"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 2) - 1)  # 归一化到[-1, 1]
    ])
    
    dataset = datasets.MNIST(
        root='../data',
        train=True,
        download=True,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # MacBook上设为0避免多进程问题
        pin_memory=False
    )
    
    return dataloader


# ==================== 主函数 ====================
def main():
    """主函数"""
    print("=" * 60)
    print("Diffusion Model Demo - MNIST数据集")
    print("=" * 60)
    
    # 创建配置
    config = Config()
    
    # 创建保存目录
    os.makedirs(config.save_dir, exist_ok=True)
    
    # 加载数据
    print("\n加载数据...")
    dataloader = get_dataloader(config)
    print(f"数据集大小: {len(dataloader.dataset)}")
    
    # 创建模型
    print("\n初始化模型...")
    diffusion = DiffusionModel(config)
    
    # 训练模型
    print("\n开始训练...")
    print(f"训练轮数: {config.epochs}")
    print(f"批次大小: {config.batch_size}")
    print(f"学习率: {config.lr}")
    diffusion.train(dataloader)
    
    # 保存模型
    print("\n保存模型...")
    diffusion.save_model()
    
    # 生成最终样本
    print("\n生成最终样本...")
    diffusion.visualize_samples('final')
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"生成的图像保存在: {config.save_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()

