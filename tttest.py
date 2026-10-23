import torch
import torch.nn as nn
import torch.nn.functional as F

# 确保CUDA可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 假设输入张量

images = torch.randn(30, 1, 68, 120, device=device)
images = images.transpose(0,1)

# 定义一个3D卷积层，时间维度的核大小为30，空间维度的核大小为3
# 确保将模型也移动到了CUDA设备上
# conv3d = nn.Conv3d(in_channels=30, out_channels=1, kernel_size=(30, 3, 3), stride=(30, 1, 1), padding=1).to(device)
conv2d = nn.Conv2d(in_channels=30, out_channels=1, kernel_size=(1, 1), stride=(1, 1)).to(device)
# 应用3D卷积
output = conv2d(images)

# 检查输出形状
print(output.shape)  # 应该输出: torch.Size([1, 512, 68, 120])