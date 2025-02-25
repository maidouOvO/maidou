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
from french_ocr import has_text, detect_text_with_language_specific_ocr
from image_type_detection import detect_image_type

def generate_summary_report(folder_a, folder_b, output_file):
    """
    生成处理结果摘要报告
    
    参数:
        folder_a: 含文字图片的文件夹路径
        folder_b: 不含文字图片的文件夹路径
        output_file: 输出报告文件路径
    """
    # 获取处理后的图片
    images_a = [f for f in os.listdir(folder_a) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images_b = [f for f in os.listdir(folder_b) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # 生成报告
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 图像处理结果摘要报告\n\n")
        
        f.write("## 处理统计\n")
        f.write(f"- 总处理图片数: {len(images_a) + len(images_b)}\n")
        f.write(f"- 含文字图片数 (文件夹A): {len(images_a)}\n")
        f.write(f"- 不含文字图片数 (文件夹B): {len(images_b)}\n\n")
        
        f.write("## 文件夹A中的图片 (含文字)\n")
        for image in images_a:
            image_path = os.path.join(folder_a, image)
            img = cv2.imread(image_path)
            if img is None:
                f.write(f"- {image}: 无法读取图片\n")
                continue
            
            # 检测图像类型
            image_type = detect_image_type(img, image)
            
            # 检测文字
            text = detect_text_with_language_specific_ocr(img, image)
            text_sample = text[:100] + "..." if len(text) > 100 else text
            
            f.write(f"- {image}:\n")
            f.write(f"  - 图像类型: {image_type}\n")
            f.write(f"  - 检测到的文字: {text_sample}\n")
            f.write(f"  - 文件大小: {os.path.getsize(image_path) / 1024:.2f} KB\n\n")
        
        f.write("## 文件夹B中的图片 (不含文字)\n")
        for image in images_b:
            image_path = os.path.join(folder_b, image)
            img = cv2.imread(image_path)
            if img is None:
                f.write(f"- {image}: 无法读取图片\n")
                continue
            
            # 检测图像类型
            image_type = detect_image_type(img, image)
            
            f.write(f"- {image}:\n")
            f.write(f"  - 图像类型: {image_type}\n")
            f.write(f"  - 文件大小: {os.path.getsize(image_path) / 1024:.2f} KB\n\n")
        
        f.write("## 处理效果分析\n")
        f.write("### 成功案例\n")
        f.write("- test_image2.png: 渐变背景图片，成功去除文字并保持渐变效果\n\n")
        
        f.write("### 需要改进的案例\n")
        f.write("- test_image1.png: 简单背景图片，文字去除效果不佳\n")
        f.write("- test_image3.png: 复杂背景图片，文字去除效果不佳\n")
        f.write("- test_image4_no_text.png: 无文字图片，被错误分类为含文字\n\n")
        
        f.write("## 改进建议\n")
        f.write("1. 优化文字检测算法，提高准确率\n")
        f.write("2. 改进文字去除算法，特别是对于复杂背景图片\n")
        f.write("3. 优化OCR文字识别，减少误判\n")
        f.write("4. 增加用户交互界面，允许手动调整文字区域\n")
        f.write("5. 添加更多语言支持，提高多语言文字识别能力\n")

if __name__ == "__main__":
    # 设置路径
    folder_a = "processed_images/folder_a"
    folder_b = "processed_images/folder_b"
    output_file = "processing_summary_report.md"
    
    # 生成报告
    generate_summary_report(folder_a, folder_b, output_file)
    print(f"报告已生成: {output_file}")
