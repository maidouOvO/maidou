#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import time

def test_text_detection_and_removal(image_path, output_dir=None):
    """
    测试文本检测和移除功能
    
    参数:
        image_path: 测试图像路径
        output_dir: 输出目录
    """
    # 创建输出目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取图像: {image_path}")
        return
    
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 使用自适应阈值处理，解决大字体空心问题
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 15, 2)
    
    # 形态学操作，填充空心区域
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建掩码
    mask = np.zeros_like(gray)
    
    # 过滤轮廓并绘制到掩码上
    for contour in contours:
        # 计算轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)
        
        # 过滤太小的轮廓
        if w < 10 or h < 10:
            continue
        
        # 计算宽高比
        aspect_ratio = w / max(h, 1)
        
        # 过滤宽高比异常的轮廓（可能是角色特征）
        if aspect_ratio < 0.1 or aspect_ratio > 10:
            continue
        
        # 提取区域
        region = gray[y:y+h, x:x+w]
        
        # 计算区域的标准差（纹理复杂度）
        std_dev = np.std(region)
        
        # 过滤纹理复杂度高的区域（可能是角色特征）
        if std_dev > 50:
            continue
        
        # 绘制轮廓到掩码上
        cv2.drawContours(mask, [contour], -1, 255, -1)
    
    # 膨胀掩码以扩大文本区域
    dilation_kernel = np.ones((7, 7), np.uint8)
    expanded_mask = cv2.dilate(mask, dilation_kernel, iterations=1)
    
    # 使用掩码移除文本
    result = cv2.inpaint(image, expanded_mask, 10, cv2.INPAINT_NS)
    
    # 保存结果
    if output_dir:
        # 保存检测结果
        cv2.imwrite(os.path.join(output_dir, "binary.jpg"), binary)
        cv2.imwrite(os.path.join(output_dir, "mask.jpg"), mask)
        cv2.imwrite(os.path.join(output_dir, "expanded_mask.jpg"), expanded_mask)
        cv2.imwrite(os.path.join(output_dir, "result.jpg"), result)
        
        # 在原图上绘制检测框
        img_draw = image.copy()
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= 10 and h >= 10:
                aspect_ratio = w / max(h, 1)
                if 0.1 <= aspect_ratio <= 10:
                    region = gray[y:y+h, x:x+w]
                    std_dev = np.std(region)
                    if std_dev <= 50:
                        cv2.rectangle(img_draw, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        cv2.imwrite(os.path.join(output_dir, "boxes.jpg"), img_draw)
    
    print("测试完成！")
    return result, expanded_mask

if __name__ == "__main__":
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='测试文本检测和移除功能')
    parser.add_argument('--image', type=str, required=True, help='测试图像路径')
    parser.add_argument('--output', type=str, default=None, help='输出目录')
    
    args = parser.parse_args()
    
    # 测试文本检测和移除功能
    test_text_detection_and_removal(args.image, args.output)
