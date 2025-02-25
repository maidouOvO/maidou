#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from PIL import Image
import pytesseract

def detect_text_by_color(image):
    """
    使用颜色特征检测文字区域
    
    参数:
        image: 输入图像
        
    返回:
        文字区域掩码
    """
    # 转换到HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 提取饱和度通道（文字通常饱和度较低）
    s_channel = hsv[:,:,1]
    
    # 使用Otsu阈值分割
    _, s_thresh = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    s_mask = cv2.morphologyEx(s_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return s_mask

def detect_text_by_color_enhanced(image):
    """
    增强版颜色特征文字检测，结合多种颜色空间特征
    
    参数:
        image: 输入图像
        
    返回:
        文字区域掩码
    """
    # 转换到HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 提取饱和度通道（文字通常饱和度较低）
    s_channel = hsv[:,:,1]
    
    # 转换到LAB颜色空间
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # 提取亮度通道（文字通常与背景亮度对比明显）
    l_channel = lab[:,:,0]
    
    # 对饱和度通道使用Otsu阈值分割
    _, s_thresh = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 对亮度通道使用自适应阈值
    l_thresh = cv2.adaptiveThreshold(l_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 11, 2)
    
    # 合并两个阈值结果
    combined_thresh = cv2.bitwise_or(s_thresh, l_thresh)
    
    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined_mask = cv2.morphologyEx(combined_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return combined_mask

def detect_text_by_color_with_edge(image):
    """
    结合颜色特征和边缘特征检测文字区域
    
    参数:
        image: 输入图像
        
    返回:
        文字区域掩码
    """
    # 转换到HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 提取饱和度通道
    s_channel = hsv[:,:,1]
    
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 边缘检测
    edges = cv2.Canny(gray, 100, 200)
    
    # 对饱和度通道使用Otsu阈值分割
    _, s_thresh = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 合并饱和度阈值和边缘检测结果
    combined = cv2.bitwise_or(s_thresh, edges)
    
    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined_mask = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return combined_mask

def test_color_detection(image_path, output_dir=None):
    """
    测试颜色检测方法
    
    参数:
        image_path: 输入图像路径
        output_dir: 输出目录，默认为None（不保存结果）
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    
    # 应用三种颜色检测方法
    mask1 = detect_text_by_color(image)
    mask2 = detect_text_by_color_enhanced(image)
    mask3 = detect_text_by_color_with_edge(image)
    
    # 如果指定了输出目录，保存结果
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.basename(image_path).split('.')[0]
        
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_mask1.png"), mask1)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_mask2.png"), mask2)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_mask3.png"), mask3)
        
        # 可视化结果
        vis1 = image.copy()
        vis2 = image.copy()
        vis3 = image.copy()
        
        # 将掩码应用到图像上
        vis1[mask1 > 0] = [0, 0, 255]  # 红色标记文字区域
        vis2[mask2 > 0] = [0, 0, 255]
        vis3[mask3 > 0] = [0, 0, 255]
        
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_vis1.png"), vis1)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_vis2.png"), vis2)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_vis3.png"), vis3)
    
    return mask1, mask2, mask3

if __name__ == "__main__":
    # 测试代码
    import os
    
    # 测试目录
    test_dir = os.path.expanduser("~/repos/maidou/image_processing/folder1")
    output_dir = os.path.expanduser("~/repos/maidou/image_processing/color_detection_results")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取测试图像
    image_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))]
    
    # 测试每个图像
    for image_path in image_files:
        print(f"处理图像: {os.path.basename(image_path)}")
        test_color_detection(image_path, output_dir)
