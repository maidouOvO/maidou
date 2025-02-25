#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import cv2
import numpy as np
from PIL import Image
import pytesseract

# 添加当前目录到系统路径，以便导入其他模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入其他模块
from advanced_inpainting import evaluate_inpaint_quality
from french_ocr import detect_text_with_language_specific_ocr
from image_type_detection import detect_image_type

def verify_text_removal_quality(original_image_path, processed_image_path):
    """
    验证文字去除质量
    
    参数:
        original_image_path: 原始图像路径
        processed_image_path: 处理后的图像路径
    """
    # 读取图像
    original_image = cv2.imread(original_image_path)
    processed_image = cv2.imread(processed_image_path)
    
    if original_image is None or processed_image is None:
        print(f"错误: 无法读取图像")
        return False
    
    # 获取文件名
    filename = os.path.basename(original_image_path)
    print(f"验证图像: {filename}")
    
    # 检测图像类型
    image_type = detect_image_type(original_image, filename)
    print(f"图像类型: {image_type}")
    
    # 创建掩码（模拟文字区域）
    # 这里使用简单的差异检测来估计文字区域
    gray_orig = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    gray_proc = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    
    # 确保尺寸相同
    if gray_orig.shape != gray_proc.shape:
        gray_proc = cv2.resize(gray_proc, (gray_orig.shape[1], gray_orig.shape[0]))
    
    # 计算差异
    diff = cv2.absdiff(gray_orig, gray_proc)
    _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # 形态学操作，连接相邻区域
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # 评估修复质量
    quality_score = evaluate_inpaint_quality(original_image, processed_image, mask)
    print(f"修复质量评分: {quality_score:.4f}")
    
    # 计算均方误差 (MSE)
    mse = np.mean((gray_orig - gray_proc) ** 2)
    print(f"均方误差 (MSE): {mse:.4f}")
    
    # 计算结构相似性指数 (SSIM)
    try:
        from skimage.metrics import structural_similarity as ssim
        ssim_score = ssim(gray_orig, gray_proc)
        print(f"结构相似性指数 (SSIM): {ssim_score:.4f}")
    except ImportError:
        print("警告: 无法导入skimage.metrics，跳过SSIM计算")
        ssim_score = None
    
    # 检测原始图像中的文字
    original_text = detect_text_with_language_specific_ocr(original_image, filename)
    print(f"原始图像文字: {original_text[:100]}...")
    
    # 检测处理后图像中的文字
    processed_text = detect_text_with_language_specific_ocr(processed_image, filename)
    print(f"处理后图像文字: {processed_text[:100]}...")
    
    # 计算文字减少百分比
    if len(original_text.strip()) > 0:
        text_reduction = 1.0 - len(processed_text.strip()) / len(original_text.strip())
        print(f"文字减少百分比: {text_reduction:.2%}")
    else:
        text_reduction = 0
        print("原始图像未检测到文字")
    
    # 根据图像类型进行特定验证
    if image_type == 'watercolor':
        # 对于水彩图像，验证LAB颜色空间处理是否保留了背景纹理
        verify_watercolor_quality(original_image, processed_image)
    elif image_type == 'gradient':
        # 对于渐变背景图像，验证渐变是否保持
        verify_gradient_quality(original_image, processed_image)
    
    # 综合评估
    quality_threshold = 0.5  # 质量评分阈值
    mse_threshold = 1000     # MSE阈值
    text_reduction_threshold = 0.5  # 文字减少百分比阈值
    
    quality_ok = quality_score > quality_threshold
    mse_ok = mse < mse_threshold
    text_reduction_ok = text_reduction > text_reduction_threshold
    
    overall_quality = quality_ok and mse_ok and text_reduction_ok
    
    print("\n综合评估:")
    print(f"修复质量: {'良好' if quality_ok else '不佳'}")
    print(f"图像保真度: {'良好' if mse_ok else '不佳'}")
    print(f"文字去除效果: {'良好' if text_reduction_ok else '不佳'}")
    print(f"总体质量: {'通过' if overall_quality else '不通过'}")
    
    return overall_quality

def verify_watercolor_quality(original_image, processed_image):
    """
    验证水彩图像的处理质量
    
    参数:
        original_image: 原始图像
        processed_image: 处理后的图像
    """
    # 转换为LAB颜色空间
    lab_orig = cv2.cvtColor(original_image, cv2.COLOR_BGR2LAB)
    lab_proc = cv2.cvtColor(processed_image, cv2.COLOR_BGR2LAB)
    
    # 提取a和b通道（颜色信息）
    a_orig = lab_orig[:,:,1]
    b_orig = lab_orig[:,:,2]
    a_proc = lab_proc[:,:,1]
    b_proc = lab_proc[:,:,2]
    
    # 计算颜色通道的相似度
    a_similarity = 1.0 - np.mean(np.abs(a_orig - a_proc)) / 255.0
    b_similarity = 1.0 - np.mean(np.abs(b_orig - b_proc)) / 255.0
    
    print("\n水彩图像质量验证:")
    print(f"a通道相似度: {a_similarity:.4f}")
    print(f"b通道相似度: {b_similarity:.4f}")
    print(f"颜色保持度: {(a_similarity + b_similarity) / 2:.4f}")
    
    # 计算纹理保持度
    gray_orig = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    gray_proc = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    
    # 使用Laplacian算子计算纹理
    texture_orig = cv2.Laplacian(gray_orig, cv2.CV_64F)
    texture_proc = cv2.Laplacian(gray_proc, cv2.CV_64F)
    
    texture_similarity = 1.0 - np.mean(np.abs(texture_orig - texture_proc)) / np.mean(np.abs(texture_orig) + 1e-10)
    print(f"纹理保持度: {texture_similarity:.4f}")

def verify_gradient_quality(original_image, processed_image):
    """
    验证渐变背景图像的处理质量
    
    参数:
        original_image: 原始图像
        processed_image: 处理后的图像
    """
    # 转换为灰度图
    gray_orig = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    gray_proc = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    
    # 计算水平和垂直方向的梯度
    sobelx_orig = cv2.Sobel(gray_orig, cv2.CV_64F, 1, 0, ksize=3)
    sobely_orig = cv2.Sobel(gray_orig, cv2.CV_64F, 0, 1, ksize=3)
    sobelx_proc = cv2.Sobel(gray_proc, cv2.CV_64F, 1, 0, ksize=3)
    sobely_proc = cv2.Sobel(gray_proc, cv2.CV_64F, 0, 1, ksize=3)
    
    # 计算梯度方向
    direction_orig = np.arctan2(sobely_orig, sobelx_orig) * 180 / np.pi
    direction_proc = np.arctan2(sobely_proc, sobelx_proc) * 180 / np.pi
    
    # 计算梯度方向的一致性
    direction_diff = np.abs(direction_orig - direction_proc)
    direction_diff = np.minimum(direction_diff, 360 - direction_diff)  # 处理角度循环
    direction_similarity = 1.0 - np.mean(direction_diff) / 180.0
    
    print("\n渐变背景质量验证:")
    print(f"梯度方向一致性: {direction_similarity:.4f}")
    
    # 计算颜色渐变的平滑度
    smoothness_orig = 1.0 / (np.std(gray_orig) + 1e-10)
    smoothness_proc = 1.0 / (np.std(gray_proc) + 1e-10)
    smoothness_ratio = smoothness_proc / smoothness_orig
    
    print(f"渐变平滑度比率: {smoothness_ratio:.4f}")
    print(f"渐变保持度: {'良好' if 0.8 < smoothness_ratio < 1.2 else '不佳'}")

if __name__ == "__main__":
    # 测试代码
    import sys
    
    if len(sys.argv) < 3:
        # 使用默认图像
        original_image_path = "test_images/test_result.png"
        processed_image_path = "processed_images/folder_a/test_result.png"
    else:
        original_image_path = sys.argv[1]
        processed_image_path = sys.argv[2]
    
    verify_text_removal_quality(original_image_path, processed_image_path)
