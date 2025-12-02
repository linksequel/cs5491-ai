"""
快速演示脚本 - 使用更少的训练轮数和时间步来快速看到效果
适合快速测试和理解diffusion模型的工作原理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# 导入主模型
import sys
sys.path.append(os.path.dirname(__file__))


# 快速配置 - 更少的参数用于演示
class QuickConfig:
    image_size = 28
    channels = 1
    batch_size = 256  # 更大的batch加速训练
    timesteps = 300  # 减少时间步
    beta_start = 0.0001
    beta_end = 0.02
    epochs = 5  # 只训练5轮
    lr = 3e-4
    device = torch.device("mps" if torch.backends.mps.is_available() 
                         else "cuda" if torch.cuda.is_available() 
                         else "cpu")
    save_dir = "quick_demo_results"
    model_path = "quick_model.pth"


def visualize_diffusion_process():
    """可视化扩散过程（前向加噪）"""
    print("\n=== 可视化扩散过程 ===")
    
    config = QuickConfig()
    
    # 加载一张MNIST图像
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 2) - 1)
    ])
    dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    image, label = dataset[0]
    print(f"原始图像标签: {label}")
    
    # 准备噪声调度
    from diffsuion_model import prepare_noise_schedule
    noise_schedule = prepare_noise_schedule(config.timesteps, config.beta_start, config.beta_end)
    
    # 展示不同时间步的图像
    timesteps_to_show = [0, 50, 100, 150, 200, 250, 299]
    fig, axes = plt.subplots(1, len(timesteps_to_show), figsize=(15, 3))
    
    noise = torch.randn_like(image)
    
    for idx, t in enumerate(timesteps_to_show):
        sqrt_alpha = noise_schedule['sqrt_alphas_cumprod'][t]
        sqrt_one_minus_alpha = noise_schedule['sqrt_one_minus_alphas_cumprod'][t]
        
        # 添加噪声
        noisy_image = sqrt_alpha * image + sqrt_one_minus_alpha * noise
        
        # 显示
        axes[idx].imshow(noisy_image.squeeze(), cmap='gray')
        axes[idx].set_title(f't={t}')
        axes[idx].axis('off')
    
    plt.tight_layout()
    os.makedirs(config.save_dir, exist_ok=True)
    plt.savefig(f'{config.save_dir}/diffusion_process.png', dpi=150, bbox_inches='tight')
    print(f"扩散过程可视化已保存到 {config.save_dir}/diffusion_process.png")
    plt.close()


def quick_train():
    """快速训练演示"""
    print("\n=== 快速训练演示 ===")
    
    from diffsuion_model import DiffusionModel, get_dataloader
    
    config = QuickConfig()
    
    # 加载数据
    print("加载数据...")
    dataloader = get_dataloader(config)
    
    # 创建模型
    print("初始化模型...")
    diffusion = DiffusionModel(config)
    
    # 训练
    print(f"\n开始快速训练 ({config.epochs} epochs)...")
    print("提示: 这个快速版本使用更少的参数来加快训练")
    diffusion.train(dataloader)
    
    # 保存
    diffusion.save_model()
    
    # 生成样本
    print("\n生成样本...")
    diffusion.visualize_samples('quick_final')
    
    return diffusion


def compare_noise_schedules():
    """比较不同的噪声调度"""
    print("\n=== 比较噪声调度 ===")
    
    from diffsuion_model import linear_beta_schedule
    
    timesteps = 300
    
    # 不同的beta范围
    schedules = [
        ("慢速", 0.0001, 0.01),
        ("中速", 0.0001, 0.02),
        ("快速", 0.0001, 0.04),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (name, beta_start, beta_end) in enumerate(schedules):
        betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        axes[idx].plot(alphas_cumprod.numpy())
        axes[idx].set_title(f'{name} (β_end={beta_end})')
        axes[idx].set_xlabel('时间步 t')
        axes[idx].set_ylabel('累积 α̅_t')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs("quick_demo_results", exist_ok=True)
    plt.savefig('quick_demo_results/noise_schedules.png', dpi=150, bbox_inches='tight')
    print("噪声调度比较已保存到 quick_demo_results/noise_schedules.png")
    plt.close()


def generate_interpolation(model=None):
    """在潜在空间中插值生成"""
    print("\n=== 潜在空间插值 ===")
    
    if model is None:
        from diffsuion_model import DiffusionModel
        config = QuickConfig()
        model = DiffusionModel(config)
        
        # 尝试加载训练好的模型
        if os.path.exists(config.model_path):
            model.load_model()
        else:
            print("未找到训练好的模型，跳过插值演示")
            return
    
    config = model.config
    device = model.device
    
    # 生成两个随机噪声
    noise1 = torch.randn((1, config.channels, config.image_size, config.image_size), device=device)
    noise2 = torch.randn((1, config.channels, config.image_size, config.image_size), device=device)
    
    # 插值
    alphas = torch.linspace(0, 1, 8)
    interpolated_images = []
    
    model.model.eval()
    with torch.no_grad():
        for alpha in tqdm(alphas, desc='插值生成'):
            # 在噪声空间插值
            noise_interp = (1 - alpha) * noise1 + alpha * noise2
            
            # 去噪
            img = noise_interp
            for i in reversed(range(0, config.timesteps)):
                t = torch.full((1,), i, device=device, dtype=torch.long)
                img = model.p_sample(img, t, i)
            
            interpolated_images.append(img.cpu())
    
    # 保存
    all_images = torch.cat(interpolated_images, dim=0)
    all_images = (all_images + 1) / 2
    save_image(all_images, f'{config.save_dir}/interpolation.png', nrow=8)
    print(f"插值结果已保存到 {config.save_dir}/interpolation.png")


def main():
    """主演示流程"""
    print("=" * 60)
    print("Diffusion Model 快速演示")
    print("=" * 60)
    
    # 1. 可视化扩散过程
    visualize_diffusion_process()
    
    # 2. 比较不同的噪声调度
    compare_noise_schedules()
    
    # 3. 快速训练
    print("\n提示: 即将开始训练，这可能需要几分钟...")
    print("如果只想看可视化效果，可以按 Ctrl+C 中断")
    
    try:
        diffusion_model = quick_train()
        
        # 4. 插值演示
        generate_interpolation(diffusion_model)
        
    except KeyboardInterrupt:
        print("\n训练已中断")
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print(f"结果保存在: quick_demo_results/")
    print("=" * 60)


if __name__ == "__main__":
    main()

