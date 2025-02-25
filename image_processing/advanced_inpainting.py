#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from PIL import Image
import os

def advanced_inpaint(image, mask):
    """
    使用高级修复算法填充文字区域
    
    参数:
        image: 输入图像
        mask: 文字区域掩码
        
    返回:
        修复后的图像
    """
    # 尝试多种修复算法并选择最佳结果
    
    # 1. TELEA算法
    result1 = cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)
    
    # 2. NS算法
    result2 = cv2.inpaint(image, mask, 5, cv2.INPAINT_NS)
    
    # 3. 扩展修复半径
    result3 = cv2.inpaint(image, mask, 10, cv2.INPAINT_NS)
    
    # 评估结果质量（使用边缘保持度量）
    quality1 = evaluate_inpaint_quality(image, result1, mask)
    quality2 = evaluate_inpaint_quality(image, result2, mask)
    quality3 = evaluate_inpaint_quality(image, result3, mask)
    
    # 返回质量最高的结果
    if quality1 >= quality2 and quality1 >= quality3:
        return result1
    elif quality2 >= quality1 and quality2 >= quality3:
        return result2
    else:
        return result3

def evaluate_inpaint_quality(original, inpainted, mask):
    """
    评估修复质量
    
    参数:
        original: 原始图像
        inpainted: 修复后的图像
        mask: 文字区域掩码
        
    返回:
        质量评分（越高越好）
    """
    # 计算非掩码区域的相似度（应该保持不变）
    inv_mask = cv2.bitwise_not(mask)
    
    # 转换为灰度图
    if len(original.shape) == 3:
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    else:
        orig_gray = original
        
    if len(inpainted.shape) == 3:
        inpaint_gray = cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY)
    else:
        inpaint_gray = inpainted
    
    # 计算非掩码区域的均方误差（MSE）
    mse_non_mask = np.sum(((orig_gray * (inv_mask / 255.0)) - (inpaint_gray * (inv_mask / 255.0))) ** 2)
    mse_non_mask = mse_non_mask / (np.sum(inv_mask / 255.0) + 1e-10)
    
    # 计算边缘保持度量
    # 对原图和修复图进行边缘检测
    edges_orig = cv2.Canny(orig_gray, 100, 200)
    edges_inpaint = cv2.Canny(inpaint_gray, 100, 200)
    
    # 在掩码区域周围创建一个扩展区域
    kernel = np.ones((5, 5), np.uint8)
    mask_dilated = cv2.dilate(mask, kernel, iterations=2)
    mask_border = cv2.subtract(mask_dilated, mask)
    
    # 计算边缘区域的相似度
    edge_similarity = np.sum((edges_orig * (mask_border / 255.0)) * (edges_inpaint * (mask_border / 255.0)))
    edge_similarity = edge_similarity / (np.sum(mask_border / 255.0) * np.sum(edges_orig * (mask_border / 255.0)) + 1e-10)
    
    # 计算纹理保持度量
    # 使用局部二值模式（LBP）特征
    texture_orig = compute_lbp(orig_gray)
    texture_inpaint = compute_lbp(inpaint_gray)
    
    # 计算掩码区域的纹理相似度
    texture_similarity = np.sum(((texture_orig * (mask / 255.0)) - (texture_inpaint * (mask / 255.0))) ** 2)
    texture_similarity = 1.0 / (texture_similarity / (np.sum(mask / 255.0) + 1e-10) + 1e-10)
    
    # 综合评分（权重可以根据需要调整）
    # 非掩码区域相似度权重高，边缘保持和纹理保持权重适中
    score = (1.0 / (mse_non_mask + 1e-10)) * 0.5 + edge_similarity * 0.3 + texture_similarity * 0.2
    
    return score

def compute_lbp(image, radius=1, n_points=8):
    """
    计算局部二值模式（LBP）特征
    
    参数:
        image: 输入灰度图像
        radius: 半径
        n_points: 采样点数
        
    返回:
        LBP特征图像
    """
    # 初始化结果图像
    height, width = image.shape
    result = np.zeros((height, width), dtype=np.uint8)
    
    # 对每个像素计算LBP值
    for y in range(radius, height - radius):
        for x in range(radius, width - radius):
            # 获取中心像素值
            center = image[y, x]
            
            # 初始化LBP值
            lbp_value = 0
            
            # 计算采样点的坐标和LBP值
            for i in range(n_points):
                # 计算采样点坐标
                theta = 2 * np.pi * i / n_points
                x_i = x + int(round(radius * np.cos(theta)))
                y_i = y + int(round(radius * np.sin(theta)))
                
                # 比较采样点与中心点的大小
                if image[y_i, x_i] >= center:
                    lbp_value += (1 << i)
            
            # 设置结果图像的值
            result[y, x] = lbp_value
    
    return result

def gradient_based_inpaint(image, mask):
    """
    基于梯度的修复算法，适用于渐变背景
    
    参数:
        image: 输入图像
        mask: 文字区域掩码
        
    返回:
        修复后的图像
    """
    # 创建结果图像
    result = image.copy()
    
    # 获取图像尺寸
    h, w = image.shape[:2]
    
    # 获取掩码区域的坐标
    y_indices, x_indices = np.where(mask > 0)
    
    # 如果没有掩码区域，直接返回原图
    if len(y_indices) == 0:
        return result
    
    # 获取四个角的颜色
    top_left = image[0, 0].astype(np.float32)
    top_right = image[0, w-1].astype(np.float32)
    bottom_left = image[h-1, 0].astype(np.float32)
    bottom_right = image[h-1, w-1].astype(np.float32)
    
    # 对掩码区域的每个像素进行插值
    for y, x in zip(y_indices, x_indices):
        # 计算相对位置
        r_ratio = x / float(w)
        c_ratio = y / float(h)
        
        # 双线性插值计算颜色
        color = (1-r_ratio)*(1-c_ratio)*top_left + \
                r_ratio*(1-c_ratio)*top_right + \
                (1-r_ratio)*c_ratio*bottom_left + \
                r_ratio*c_ratio*bottom_right
        
        # 设置结果图像的值
        result[y, x] = color.astype(np.uint8)
    
    return result

def test_inpainting(image_path, output_dir=None):
    """
    测试修复算法
    
    参数:
        image_path: 输入图像路径
        output_dir: 输出目录，默认为None（不保存结果）
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    
    # 创建掩码（模拟文字区域）
    # 这里使用简单的矩形区域作为示例
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    h, w = image.shape[:2]
    cv2.rectangle(mask, (int(w*0.3), int(h*0.4)), (int(w*0.7), int(h*0.6)), 255, -1)
    
    # 应用修复算法
    result1 = cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)
    result2 = cv2.inpaint(image, mask, 5, cv2.INPAINT_NS)
    result3 = cv2.inpaint(image, mask, 10, cv2.INPAINT_NS)
    result4 = gradient_based_inpaint(image, mask)
    result5 = advanced_inpaint(image, mask)
    
    # 如果指定了输出目录，保存结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.basename(image_path).split('.')[0]
        
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_mask.png"), mask)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_telea.png"), result1)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_ns.png"), result2)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_ns_extended.png"), result3)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_gradient.png"), result4)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_advanced.png"), result5)
    
    return result1, result2, result3, result4, result5

if __name__ == "__main__":
    # 测试代码
    import os
    
    # 测试目录
    test_dir = os.path.expanduser("~/repos/maidou/image_processing/folder1")
    output_dir = os.path.expanduser("~/repos/maidou/image_processing/inpainting_results")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取测试图像
    image_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))]
    
    # 测试每个图像
    for image_path in image_files:
        print(f"处理图像: {os.path.basename(image_path)}")
        test_inpainting(image_path, output_dir)
