import os
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from craft import CRAFT
from craft_utils import getDetBoxes, adjustResultCoordinates
from imgproc import normalizeMeanVariance, resize_aspect_ratio

class TextDetector:
    def __init__(self, model_path='models/craft_mlt_25k.pth', cuda=False):
        self.cuda = cuda
        self.model_path = model_path
        self.net = CRAFT()
        
        if cuda:
            state_dict = torch.load(model_path)
            self.net = self.net.cuda()
            cudnn.benchmark = False
        else:
            state_dict = torch.load(model_path, map_location='cpu')
            
        # 处理模型键名前缀问题
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        self.net.load_state_dict(new_state_dict)
        
        self.net.eval()
    
    def detect(self, image, text_threshold=0.7, link_threshold=0.4, low_text=0.4, 
               canvas_size=1280, mag_ratio=1.5, poly=True):  # 默认使用多边形
        # 预处理图像
        img_resized, target_ratio, size_heatmap = resize_aspect_ratio(image, 
                                                                      canvas_size, 
                                                                      interpolation=cv2.INTER_LINEAR, 
                                                                      mag_ratio=mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio
        
        # 归一化
        x = normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        x = x.unsqueeze(0)                          # [c, h, w] to [b, c, h, w]
        
        # 前向传播
        with torch.no_grad():
            if self.cuda:
                x = x.cuda()
            y, _ = self.net(x)
        
        # 后处理
        score_text = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()
        
        # 获取检测框
        boxes, polys = getDetBoxes(score_text, score_link, text_threshold, 
                                  link_threshold, low_text, poly)
        
        # 调整坐标
        boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
        
        # 打印调试信息
        print(f"检测到的文本框数量: {len(boxes)}")
        print(f"检测到的多边形数量: {len(polys)}")
        
        return boxes, polys
