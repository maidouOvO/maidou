#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os

def analyze_image(image_path):
    """分析图片特性"""
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return
    
    # 基本信息
    print(f"图片路径: {image_path}")
    print(f"图片尺寸: {img.shape}")
    print(f"图片类型: {img.dtype}")
    print(f"像素值范围: {np.min(img)} - {np.max(img)}")
    
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 计算直方图
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    print(f"灰度直方图峰值: {np.argmax(hist)}")
    
    # 边缘检测
    edges = cv2.Canny(gray, 100, 200)
    edge_pixels = np.count_nonzero(edges)
    print(f"边缘像素数量: {edge_pixels}")
    print(f"边缘像素占比: {edge_pixels / (img.shape[0] * img.shape[1]) * 100:.2f}%")
    
    # 尝试不同的阈值方法
    print("\n不同阈值方法的效果:")
    
    # 全局阈值
    _, thresh_global = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # 自适应阈值 - 高斯
    thresh_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY_INV, 11, 2)
    
    # Otsu阈值
    _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 计算每种方法检测到的"前景"像素数量
    print(f"全局阈值前景像素: {np.count_nonzero(thresh_global)}")
    print(f"自适应阈值前景像素: {np.count_nonzero(thresh_adaptive)}")
    print(f"Otsu阈值前景像素: {np.count_nonzero(thresh_otsu)}")
    
    # MSER区域检测
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)
    print(f"MSER检测到的区域数量: {len(regions)}")
    
    # 保存分析结果
    output_dir = os.path.dirname(image_path) + "_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.basename(image_path).split('.')[0]
    
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_gray.png"), gray)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_edges.png"), edges)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_thresh_global.png"), thresh_global)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_thresh_adaptive.png"), thresh_adaptive)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_thresh_otsu.png"), thresh_otsu)
    
    # 绘制MSER区域
    mser_vis = img.copy()
    for region in regions:
        hull = cv2.convexHull(region.reshape(-1, 1, 2))
        cv2.polylines(mser_vis, [hull], True, (0, 255, 0), 2)
    
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_mser.png"), mser_vis)
    
    print(f"分析结果已保存到: {output_dir}")

if __name__ == "__main__":
    # 分析问题图片
    analyze_image(os.path.expanduser("~/image_processing_project/folder1/test_image3.png"))
