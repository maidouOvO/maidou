#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from collections import OrderedDict
import sys

# 添加当前目录和CRAFT-pytorch目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

craft_dir = os.path.join(current_dir, 'CRAFT-pytorch')
if craft_dir not in sys.path:
    sys.path.append(craft_dir)

# 导入CRAFT模型和工具
from CRAFT_pytorch.craft import CRAFT
from CRAFT_pytorch.craft_utils import getDetBoxes, adjustResultCoordinates
from CRAFT_pytorch.imgproc import resize_aspect_ratio, normalizeMeanVariance

def copyStateDict(state_dict):
    """复制模型状态字典"""
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def load_craft_model(weights_path, cuda=False):
    """
    加载CRAFT模型
    
    参数:
        weights_path: 权重文件路径
        cuda: 是否使用CUDA
        
    返回:
        net: 加载的模型
    """
    # 初始化模型
    net = CRAFT()
    
    print(f'正在加载预训练模型: {weights_path}')
    if cuda:
        net.load_state_dict(copyStateDict(torch.load(weights_path)))
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False
    else:
        net.load_state_dict(copyStateDict(torch.load(weights_path, map_location='cpu')))
    
    net.eval()
    return net

def detect_text_regions(net, image, text_threshold=0.7, link_threshold=0.4, low_text=0.4, 
                        cuda=False, poly=False, refine_net=None, show_time=False,
                        canvas_size=1280, mag_ratio=1.5):
    """
    检测图像中的文本区域
    
    参数:
        net: CRAFT模型
        image: 输入图像
        text_threshold: 文本置信度阈值
        link_threshold: 链接置信度阈值
        low_text: 低文本置信度阈值
        cuda: 是否使用CUDA
        poly: 是否使用多边形
        refine_net: 精细化网络
        show_time: 是否显示处理时间
        canvas_size: 画布大小
        mag_ratio: 放大比例
        
    返回:
        boxes: 检测到的文本框
        polys: 检测到的文本多边形
        ret_score_text: 文本得分图
    """
    t0 = time.time()
    
    # 调整图像大小
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(image, canvas_size, 
                                                                interpolation=cv2.INTER_LINEAR, 
                                                                mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio
    
    # 预处理图像
    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = x.unsqueeze(0)                          # [c, h, w] to [b, c, h, w]
    
    # 使用CUDA（如果可用）
    if cuda:
        x = x.cuda()
    
    # 前向传播
    with torch.no_grad():
        y, feature = net(x)
    
    # 获取输出
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()
    
    # 获取文本框
    boxes, labels, mapper = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)
    
    # 调整坐标
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = boxes
    
    # 过滤太小的文本框
    for k in range(len(polys)):
        if polys[k] is not None:
            if len(polys[k]) < 4:
                polys[k] = None
    
    t1 = time.time()
    
    if show_time:
        print(f"文本检测耗时: {t1 - t0:.3f}s")
    
    return boxes, polys, score_text, labels

def get_character_boxes(polys, image, labels=None):
    """
    获取字符级别的边界框
    
    参数:
        polys: 检测到的文本多边形
        image: 原始图像
        labels: 标签图像
        
    返回:
        char_boxes: 字符级别的边界框
    """
    char_boxes = []
    
    for i, poly in enumerate(polys):
        if poly is None:
            continue
        
        # 计算多边形的边界框
        min_x = min([p[0] for p in poly])
        min_y = min([p[1] for p in poly])
        max_x = max([p[0] for p in poly])
        max_y = max([p[1] for p in poly])
        
        # 确保坐标在图像范围内
        min_x = max(0, int(min_x))
        min_y = max(0, int(min_y))
        max_x = min(image.shape[1] - 1, int(max_x))
        max_y = min(image.shape[0] - 1, int(max_y))
        
        # 提取区域
        region = image[min_y:max_y, min_x:max_x]
        
        if region.size == 0:
            continue
        
        # 转换为灰度图
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # 使用自适应阈值处理，解决大字体空心问题
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 15, 2)
        
        # 形态学操作，填充空心区域
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤轮廓
        for contour in contours:
            # 计算轮廓的边界框
            x, y, w, h = cv2.boundingRect(contour)
            
            # 过滤太小的轮廓
            if w < 5 or h < 5:
                continue
            
            # 转换回原始图像坐标
            x1 = min_x + x
            y1 = min_y + y
            x2 = x1 + w
            y2 = y1 + h
            
            char_boxes.append([x1, y1, x2, y2])
    
    return char_boxes

def create_refined_text_mask(image, text_polys, char_boxes=None, dilation_size=7, use_char_level=True):
    """
    创建精细化的文本遮罩
    
    参数:
        image: 输入图像
        text_polys: 文本多边形区域列表
        char_boxes: 字符级别边界框
        dilation_size: 膨胀核大小
        use_char_level: 是否使用字符级别检测
        
    返回:
        refined_mask: 精细化的文本遮罩
    """
    # 创建初始掩码
    initial_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # 在掩码上绘制文本区域
    for poly in text_polys:
        if poly is not None:
            # 将多边形转换为整数坐标
            poly_points = np.array(poly, dtype=np.int32)
            cv2.fillPoly(initial_mask, [poly_points], 255)  # 填充多边形区域
    
    # 膨胀掩码以扩大文本区域
    kernel = np.ones((dilation_size, dilation_size), np.uint8)
    expanded_mask = cv2.dilate(initial_mask, kernel, iterations=1)
    
    # 创建精细化掩码
    refined_mask = np.zeros_like(expanded_mask)
    
    # 如果使用字符级别检测
    if use_char_level and char_boxes is not None and len(char_boxes) > 0:
        # 在精细化掩码上绘制字符区域
        for box in char_boxes:
            x1, y1, x2, y2 = [int(coord) for coord in box]
            cv2.rectangle(refined_mask, (x1, y1), (x2, y2), 255, -1)
    else:
        # 否则使用扩展掩码
        refined_mask = expanded_mask
    
    return refined_mask

def is_character_feature(box, image, aspect_ratio_threshold=0.1, area_threshold=100):
    """
    判断检测到的区域是否为角色特征（如眼睛、嘴巴等）而非文本
    
    参数:
        box: 检测到的边界框 [x1, y1, x2, y2]
        image: 原始图像
        aspect_ratio_threshold: 宽高比阈值
        area_threshold: 面积阈值
        
    返回:
        is_feature: 是否为角色特征
    """
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    
    # 计算宽高比
    aspect_ratio = width / max(height, 1)
    
    # 计算面积
    area = width * height
    
    # 提取区域
    region = image[y1:y2, x1:x2]
    
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    else:
        gray = region
    
    # 计算区域的标准差（纹理复杂度）
    std_dev = np.std(gray)
    
    # 判断是否为角色特征
    # 1. 宽高比过小或过大（非常窄或非常宽）
    # 2. 面积过小
    # 3. 纹理复杂度高
    if aspect_ratio < aspect_ratio_threshold or aspect_ratio > 1/aspect_ratio_threshold:
        return True
    if area < area_threshold:
        return True
    if std_dev > 50:  # 高纹理复杂度
        return True
    
    return False

def filter_text_regions(boxes, polys, image, preserve_character_features=True):
    """
    过滤文本区域，移除可能是角色特征的区域
    
    参数:
        boxes: 检测到的文本框
        polys: 检测到的文本多边形
        image: 原始图像
        preserve_character_features: 是否保留角色特征
        
    返回:
        filtered_boxes: 过滤后的文本框
        filtered_polys: 过滤后的文本多边形
    """
    if not preserve_character_features:
        return boxes, polys
    
    filtered_boxes = []
    filtered_polys = []
    
    for i, (box, poly) in enumerate(zip(boxes, polys)):
        if poly is None:
            continue
        
        # 计算多边形的边界框
        min_x = min([p[0] for p in poly])
        min_y = min([p[1] for p in poly])
        max_x = max([p[0] for p in poly])
        max_y = max([p[1] for p in poly])
        
        # 确保坐标在图像范围内
        min_x = max(0, int(min_x))
        min_y = max(0, int(min_y))
        max_x = min(image.shape[1] - 1, int(max_x))
        max_y = min(image.shape[0] - 1, int(max_y))
        
        # 判断是否为角色特征
        if is_character_feature([min_x, min_y, max_x, max_y], image):
            continue
        
        filtered_boxes.append(box)
        filtered_polys.append(poly)
    
    return filtered_boxes, filtered_polys

def detect_and_create_mask(image, craft_net, text_threshold=0.7, link_threshold=0.4, low_text=0.4,
                          cuda=False, canvas_size=1280, mag_ratio=1.5, 
                          dilation_size=7, use_char_level=True, preserve_character_features=True):
    """
    检测文本并创建掩码
    
    参数:
        image: 输入图像
        craft_net: CRAFT模型
        text_threshold: 文本置信度阈值
        link_threshold: 链接置信度阈值
        low_text: 低文本置信度阈值
        cuda: 是否使用CUDA
        canvas_size: 画布大小
        mag_ratio: 放大比例
        dilation_size: 膨胀核大小
        use_char_level: 是否使用字符级别检测
        preserve_character_features: 是否保留角色特征
        
    返回:
        mask: 文本掩码
        text_polys: 文本多边形
    """
    # 检测文本区域
    boxes, polys, _, _ = detect_text_regions(
        craft_net, image, 
        text_threshold=text_threshold, 
        link_threshold=link_threshold, 
        low_text=low_text,
        cuda=cuda,
        canvas_size=canvas_size,
        mag_ratio=mag_ratio
    )
    
    # 过滤文本区域，移除可能是角色特征的区域
    if preserve_character_features:
        boxes, polys = filter_text_regions(boxes, polys, image, preserve_character_features)
    
    # 获取字符级别的边界框
    char_boxes = get_character_boxes(polys, image)
    
    # 创建精细化的文本遮罩
    mask = create_refined_text_mask(
        image, polys, char_boxes, 
        dilation_size=dilation_size, 
        use_char_level=use_char_level
    )
    
    return mask, polys
