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

def verify_all_images(folder_a, folder_b, test_folder):
    """
    验证所有处理过的图片
    
    参数:
        folder_a: 含文字图片的文件夹路径
        folder_b: 不含文字图片的文件夹路径
        test_folder: 测试图片文件夹路径
    """
    print("开始验证所有处理过的图片...")
    
    # 获取所有测试图片
    test_images = []
    for root, _, files in os.walk(test_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                test_images.append(os.path.join(root, file))
    
    # 获取处理后的图片
    processed_images_a = [os.path.join(folder_a, f) for f in os.listdir(folder_a) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    processed_images_b = [os.path.join(folder_b, f) for f in os.listdir(folder_b) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # 统计处理情况
    total_test_images = len(test_images)
    total_processed_images = len(processed_images_a) + len(processed_images_b)
    
    print(f"测试图片总数: {total_test_images}")
    print(f"处理后图片总数: {total_processed_images}")
    print(f"文件夹A (含文字) 图片数: {len(processed_images_a)}")
    print(f"文件夹B (不含文字) 图片数: {len(processed_images_b)}")
    
    # 验证分类是否正确
    print("\n验证图片分类...")
    
    # 检查文件夹A中的图片
    print("\n文件夹A (含文字) 中的图片:")
    for image_path in processed_images_a:
        filename = os.path.basename(image_path)
        image = cv2.imread(image_path)
        if image is None:
            print(f"  - {filename}: 无法读取图片")
            continue
        
        # 检测文字
        has_text_result = has_text(image, filename)
        print(f"  - {filename}: {'正确分类' if has_text_result else '错误分类'} (OCR检测: {'含有文字' if has_text_result else '不含文字'})")
    
    # 检查文件夹B中的图片
    print("\n文件夹B (不含文字) 中的图片:")
    for image_path in processed_images_b:
        filename = os.path.basename(image_path)
        image = cv2.imread(image_path)
        if image is None:
            print(f"  - {filename}: 无法读取图片")
            continue
        
        # 检测文字
        has_text_result = has_text(image, filename)
        print(f"  - {filename}: {'错误分类' if has_text_result else '正确分类'} (OCR检测: {'含有文字' if has_text_result else '不含文字'})")
    
    # 验证特殊图片处理
    print("\n验证特殊图片处理...")
    
    # 检查渐变背景图片
    gradient_images = [img for img in processed_images_a + processed_images_b 
                      if detect_image_type(cv2.imread(img), os.path.basename(img)) == 'gradient']
    print(f"渐变背景图片: {len(gradient_images)} 张")
    for image_path in gradient_images:
        filename = os.path.basename(image_path)
        print(f"  - {filename}")
    
    # 检查复杂背景图片
    complex_images = [img for img in processed_images_a + processed_images_b 
                     if detect_image_type(cv2.imread(img), os.path.basename(img)) == 'complex']
    print(f"复杂背景图片: {len(complex_images)} 张")
    for image_path in complex_images:
        filename = os.path.basename(image_path)
        print(f"  - {filename}")
    
    # 检查水彩背景图片
    watercolor_images = [img for img in processed_images_a + processed_images_b 
                        if detect_image_type(cv2.imread(img), os.path.basename(img)) == 'watercolor']
    print(f"水彩背景图片: {len(watercolor_images)} 张")
    for image_path in watercolor_images:
        filename = os.path.basename(image_path)
        print(f"  - {filename}")
    
    # 总结
    print("\n验证结果总结:")
    print(f"- 总测试图片数: {total_test_images}")
    print(f"- 总处理图片数: {total_processed_images}")
    print(f"- 文件夹A (含文字) 图片数: {len(processed_images_a)}")
    print(f"- 文件夹B (不含文字) 图片数: {len(processed_images_b)}")
    print(f"- 渐变背景图片数: {len(gradient_images)}")
    print(f"- 复杂背景图片数: {len(complex_images)}")
    print(f"- 水彩背景图片数: {len(watercolor_images)}")
    
    print("\n验证完成。")

if __name__ == "__main__":
    # 设置路径
    folder_a = "processed_images/folder_a"
    folder_b = "processed_images/folder_b"
    test_folder = "test_images"
    
    # 验证所有图片
    verify_all_images(folder_a, folder_b, test_folder)
