"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import numpy as np
import cv2

def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    """
    标准化图像的均值和方差
    
    Args:
        in_img: 输入图像
        mean: 均值
        variance: 方差
        
    Returns:
        标准化后的图像
    """
    # 应该是RGB顺序
    img = in_img.copy().astype(np.float32)

    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img

def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1):
    """
    调整图像大小，保持纵横比
    
    Args:
        img: 输入图像
        square_size: 目标尺寸
        interpolation: 插值方法
        mag_ratio: 放大比例
        
    Returns:
        调整大小后的图像，目标比例，热图大小
    """
    height, width, channel = img.shape

    # 放大图像
    target_size = mag_ratio * max(height, width)

    # 设置目标尺寸
    if target_size > square_size:
        target_size = square_size
    
    ratio = target_size / max(height, width)    

    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation = interpolation)
    
    # 确保大小是32的倍数
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc
    target_h, target_w = target_h32, target_w32

    size_heatmap = (int(target_w/2), int(target_h/2))

    return resized, ratio, size_heatmap

def cvt2HeatmapImg(img):
    """
    将图像转换为热图
    
    Args:
        img: 输入图像
        
    Returns:
        热图图像
    """
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img
