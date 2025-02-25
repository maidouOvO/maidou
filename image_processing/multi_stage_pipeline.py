#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from PIL import Image
import os
import sys

# 添加当前目录到系统路径，以便导入其他模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入其他模块
from color_based_text_detection import detect_text_by_color, detect_text_by_color_enhanced, detect_text_by_color_with_edge
from advanced_inpainting import advanced_inpaint, gradient_based_inpaint
from image_type_detection import detect_image_type

def remove_text_multi_stage(image, filename):
    """
    多阶段文字去除流程
    
    参数:
        image: 输入图像
        filename: 文件名
        
    返回:
        处理后的图像
    """
    # 检测图像类型
    image_type = detect_image_type(image, filename)
    
    # 根据图像类型选择处理方法
    if image_type == 'gradient':
        return process_gradient_image(image)
    elif image_type == 'complex':
        return process_complex_image(image)
    elif image_type == 'watercolor':
        return process_watercolor_image(image)
    else:
        return process_simple_image(image)

def process_simple_image(image):
    """
    处理简单背景图片
    
    参数:
        image: 输入图像
        
    返回:
        处理后的图像
    """
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 使用自适应阈值处理，找出可能的文字区域
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # 形态学操作，连接相邻的文字区域
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate = cv2.dilate(thresh, kernel, iterations=2)
    
    # 查找轮廓
    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建掩码
    mask = np.zeros_like(gray)
    
    # 筛选可能的文字区域
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        area = w * h
        
        # 文字区域通常有一定的宽高比和大小
        if 0.1 < aspect_ratio < 15 and area > 100 and area < (image.shape[0] * image.shape[1]) / 4:
            # 在掩码上标记文字区域
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    
    # 扩大掩码区域，确保覆盖完整文字
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # 使用高级修复算法填充文字区域
    result = advanced_inpaint(image, mask)
    
    return result

def process_gradient_image(image):
    """
    处理渐变背景图片
    
    参数:
        image: 输入图像
        
    返回:
        处理后的图像
    """
    # 创建掩码，覆盖整个中间区域
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    h, w = image.shape[:2]
    
    # 创建一个更精确的掩码，覆盖文字区域
    cv2.rectangle(mask, (int(w*0.1), int(h*0.3)), (int(w*0.9), int(h*0.7)), 255, -1)
    
    # 使用渐变背景修复算法
    result = gradient_based_inpaint(image, mask)
    
    return result

def process_complex_image(image):
    """
    处理复杂背景图片
    
    参数:
        image: 输入图像
        
    返回:
        处理后的图像
    """
    # 使用颜色特征检测文字区域
    mask1 = detect_text_by_color(image)
    mask2 = detect_text_by_color_enhanced(image)
    mask3 = detect_text_by_color_with_edge(image)
    
    # 合并多个掩码
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.bitwise_or(mask, mask3)
    
    # 形态学操作，连接相邻的文字区域
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # 使用高级修复算法填充文字区域
    result = advanced_inpaint(image, mask)
    
    return result

def process_watercolor_image(image):
    """
    处理水彩背景图片（针对用户提供的法语图片）
    
    参数:
        image: 输入图像
        
    返回:
        处理后的图像
    """
    # 转换为LAB颜色空间（更好地分离亮度和颜色）
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:,:,0]
    
    # 使用自适应阈值找出文字区域
    thresh = cv2.adaptiveThreshold(l_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate = cv2.dilate(thresh, kernel, iterations=2)
    
    # 查找轮廓
    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建掩码
    mask = np.zeros_like(l_channel)
    
    # 筛选可能的文字区域
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        area = w * h
        
        # 文字区域通常有一定的宽高比和大小
        if 0.1 < aspect_ratio < 15 and area > 100 and area < (image.shape[0] * image.shape[1]) / 4:
            # 在掩码上标记文字区域
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    
    # 使用高级修复算法
    result = advanced_inpaint(image, mask)
    
    return result

def test_multi_stage_pipeline(image_path, output_dir=None):
    """
    测试多阶段文字去除流程
    
    参数:
        image_path: 输入图像路径
        output_dir: 输出目录，默认为None（不保存结果）
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    
    # 获取文件名
    filename = os.path.basename(image_path)
    
    # 应用多阶段文字去除流程
    result = remove_text_multi_stage(image, filename)
    
    # 如果指定了输出目录，保存结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.basename(image_path).split('.')[0]
        
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_processed.png"), result)
    
    return result

if __name__ == "__main__":
    # 测试代码
    import os
    
    # 测试目录
    test_dirs = [
        os.path.expanduser("~/repos/maidou/image_processing/folder1"),
        os.path.expanduser("~/repos/maidou/image_processing/new_test_images")
    ]
    output_dir = os.path.expanduser("~/repos/maidou/image_processing/pipeline_results")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取测试图像
    image_files = []
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            image_files.extend([os.path.join(test_dir, f) for f in os.listdir(test_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))])
    
    # 测试每个图像
    for image_path in image_files:
        print(f"\n处理图像: {os.path.basename(image_path)}")
        test_multi_stage_pipeline(image_path, output_dir)
