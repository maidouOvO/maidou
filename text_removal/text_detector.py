import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from collections import OrderedDict

from craft.craft import CRAFT
from craft.craft_utils import getDetBoxes, adjustResultCoordinates
from craft.imgproc import resize_aspect_ratio, normalizeMeanVariance, cvt2HeatmapImg

class TextDetector:
    def __init__(self, 
                 trained_model='craft/weights/craft_mlt_25k.pth',
                 text_threshold=0.7,
                 low_text=0.4,
                 link_threshold=0.4,
                 cuda=False,
                 canvas_size=1280,
                 mag_ratio=1.5,
                 poly=False):
        """
        初始化CRAFT文本检测器
        
        Args:
            trained_model: 预训练模型路径
            text_threshold: 文本置信度阈值
            low_text: 文本低边界分数
            link_threshold: 链接置信度阈值
            cuda: 是否使用CUDA进行推理
            canvas_size: 推理图像大小
            mag_ratio: 图像放大比例
            poly: 是否启用多边形类型
        """
        self.text_threshold = text_threshold
        self.low_text = low_text
        self.link_threshold = link_threshold
        self.cuda = cuda
        self.canvas_size = canvas_size
        self.mag_ratio = mag_ratio
        self.poly = poly
        
        # 初始化网络
        self.net = CRAFT()
        
        # 加载预训练模型
        if os.path.isfile(trained_model):
            print(f'Loading weights from checkpoint ({trained_model})')
            if self.cuda:
                self.net.load_state_dict(self._copy_state_dict(torch.load(trained_model)))
                self.net = self.net.cuda()
                self.net = torch.nn.DataParallel(self.net)
                cudnn.benchmark = False
            else:
                self.net.load_state_dict(self._copy_state_dict(torch.load(trained_model, map_location='cpu')))
            
            self.net.eval()
        else:
            print(f"No checkpoint found at '{trained_model}'")
    
    def _copy_state_dict(self, state_dict):
        """
        复制模型状态字典
        """
        if list(state_dict.keys())[0].startswith("module"):
            start_idx = 1
        else:
            start_idx = 0
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = ".".join(k.split(".")[start_idx:])
            new_state_dict[name] = v
        return new_state_dict
    
    def detect(self, image):
        """
        检测图像中的文本区域
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            boxes: 文本边界框
            polys: 文本多边形
            score_text: 文本得分热图
        """
        # 调整图像大小
        img_resized, target_ratio, _ = resize_aspect_ratio(image, 
                                                          self.canvas_size, 
                                                          interpolation=cv2.INTER_LINEAR, 
                                                          mag_ratio=self.mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio
        
        # 预处理
        x = normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
        if self.cuda:
            x = x.cuda()
        
        # 前向传播
        with torch.no_grad():
            y, _ = self.net(x)
        
        # 生成得分图和链接图
        score_text = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()
        
        # 后处理
        boxes, polys = getDetBoxes(score_text, score_link, 
                                  self.text_threshold, 
                                  self.link_threshold, 
                                  self.low_text, 
                                  self.poly)
        
        # 坐标调整
        boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None: 
                polys[k] = boxes[k]
        
        # 渲染结果
        render_img = score_text.copy()
        render_img = np.hstack((render_img, score_link))
        ret_score_text = cvt2HeatmapImg(render_img)
        
        return boxes, polys, ret_score_text
