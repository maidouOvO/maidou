import numpy as np
import cv2

def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1):
    """保持宽高比调整图像大小"""
    height, width, channel = img.shape

    # 计算目标尺寸
    target_size = mag_ratio * max(height, width)
    
    # 如果目标尺寸大于square_size，则调整为square_size
    if target_size > square_size:
        target_size = square_size
    
    # 计算比例
    ratio = target_size / max(height, width)
    
    # 调整图像大小
    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)
    
    # 计算填充大小
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    
    # 创建填充后的图像
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc
    
    # 计算目标比例和热图大小
    target_ratio = target_h / height
    size_heatmap = (int(target_w32 / 2), int(target_h32 / 2))
    
    return resized, target_ratio, size_heatmap

def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    """标准化图像的均值和方差"""
    # 转换为浮点型
    img = in_img.copy().astype(np.float32)
    
    # 标准化
    img /= 255.0
    img -= mean
    img /= variance
    
    return img
