# Diffusion Model Demo

这是一个在MNIST数据集上训练的简单扩散模型(DDPM)演示。

## 什么是Diffusion Model？

扩散模型是一种生成模型，其灵感来自热力学中的扩散过程：

1. **前向过程（加噪）**: 逐步向图像添加高斯噪声，最终变成纯噪声
2. **反向过程（去噪）**: 训练神经网络学习逆向这个过程，从噪声中恢复图像

## 模型架构

- **UNet**: 使用UNet架构作为噪声预测网络
- **时间编码**: 使用正弦位置编码来编码时间步信息
- **训练目标**: 预测每个时间步添加的噪声

## 环境要求

```bash
pip install torch torchvision numpy matplotlib tqdm
```

## 使用方法

### 直接运行训练

```bash
cd week11
python diffsuion-model.py
```

### 主要参数配置

在代码中的`Config`类可以调整以下参数：

- `timesteps`: 扩散步数（默认1000）
- `epochs`: 训练轮数（默认20）
- `batch_size`: 批次大小（默认128）
- `lr`: 学习率（默认2e-4）

### 训练过程

程序会：
1. 自动下载MNIST数据集
2. 训练diffusion模型
3. 每5个epoch保存一次生成样本
4. 保存训练好的模型权重

### 输出文件

- `diffusion_results/`: 存放生成的图像
  - `samples_epoch_X.png`: 每个epoch的生成样本
  - `process_epoch_X.png`: 去噪过程的中间步骤
- `diffusion_model.pth`: 训练好的模型权重

## 设备支持

代码会自动检测可用设备：
- **MacBook (M1/M2/M3)**: 使用MPS加速
- **NVIDIA GPU**: 使用CUDA加速
- **其他**: 使用CPU

## 训练时间估计

在不同设备上的训练时间（20 epochs）：
- MacBook M1/M2: ~15-20分钟
- CPU: ~1-2小时

## 生成效果

训练完成后，模型能够从随机噪声生成手写数字图像。查看`diffusion_results/`文件夹中的图像来观察效果。

## 算法原理

### 前向扩散过程

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$$

其中：
- $x_0$: 原始图像
- $x_t$: 时间步$t$的噪声图像
- $\epsilon \sim \mathcal{N}(0, I)$: 高斯噪声
- $\bar{\alpha}_t$: 累积噪声系数

### 反向去噪过程

训练神经网络$\epsilon_\theta(x_t, t)$来预测噪声，然后使用以下公式去噪：

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right) + \sigma_t z$$

## 参考文献

- [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
- [Understanding Diffusion Models (Luo, 2022)](https://arxiv.org/abs/2208.11970)

## 扩展建议

1. **更复杂的数据集**: 尝试CIFAR-10或CelebA
2. **改进架构**: 使用Attention机制
3. **更好的采样器**: 实现DDIM等快速采样方法
4. **条件生成**: 添加类别条件或文本条件

## 常见问题

**Q: 训练很慢怎么办？**
- 减少`timesteps`到500
- 减少`epochs`到10
- 减小`batch_size`

**Q: 生成效果不好？**
- 增加训练轮数
- 调整学习率
- 检查数据预处理是否正确

**Q: MacBook上MPS不工作？**
- 确保PyTorch版本 >= 1.12
- 更新到最新的macOS版本

