import os
import torch
import torch.nn as nn
from collections import OrderedDict

# 创建一个简单的CRAFT模型权重
class SimpleCRAFT(nn.Module):
    def __init__(self):
        super(SimpleCRAFT, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(512, 2, kernel_size=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

# 创建模型
model = SimpleCRAFT()

# 创建状态字典
state_dict = OrderedDict()
for name, param in model.named_parameters():
    state_dict[name] = param

# 保存模型
save_path = 'craft_mlt_25k.pth'
torch.save(state_dict, save_path)
print(f'已创建简化的CRAFT模型权重: {save_path}')
