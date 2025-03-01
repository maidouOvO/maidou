import torch
import torch.nn as nn
from torchvision import models
from collections import namedtuple

def init_weights(modules):
    """初始化模型权重"""
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class vgg16_bn(nn.Module):
    def __init__(self, pretrained=True, freeze=True):
        super(vgg16_bn, self).__init__()
        
        # 加载预训练的VGG16模型
        vgg_pretrained_features = models.vgg16_bn(weights='DEFAULT' if pretrained else None).features
        
        # 定义各个阶段
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        
        # 将VGG16的特征层分配到各个阶段
        # 获取特征层的长度
        vgg_length = len(vgg_pretrained_features)
        
        # 根据实际长度调整切片范围
        slice1_end = min(12, vgg_length)
        slice2_end = min(19, vgg_length)
        slice3_end = min(29, vgg_length)
        slice4_end = min(39, vgg_length)
        slice5_end = vgg_length
        
        for x in range(slice1_end):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(slice1_end, slice2_end):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(slice2_end, slice3_end):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(slice3_end, slice4_end):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(slice4_end, slice5_end):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        
        # 如果freeze为True，则冻结参数
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        h = self.slice1(x)
        h_relu2_2 = h
        h = self.slice2(h)
        h_relu3_3 = h
        h = self.slice3(h)
        h_relu4_3 = h
        h = self.slice4(h)
        h_relu5_3 = h
        
        vgg_outputs = namedtuple("VggOutputs", ['fc7', 'relu5_3', 'relu4_3', 'relu3_3', 'relu2_2'])
        out = vgg_outputs(h, h_relu5_3, h_relu4_3, h_relu3_3, h_relu2_2)
        
        return out
