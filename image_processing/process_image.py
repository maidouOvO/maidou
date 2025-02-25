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
from multi_stage_pipeline import remove_text_multi_stage
from french_ocr import has_text

def process_image(input_image, output_folder_a, output_folder_b):
    """
    处理单个图像，去除文字并根据OCR结果分类
    
    参数:
        input_image: 输入图像路径
        output_folder_a: 含文字图片的输出文件夹路径
        output_folder_b: 不含文字图片的输出文件夹路径
    """
    # 确保输出目录存在
    os.makedirs(output_folder_a, exist_ok=True)
    os.makedirs(output_folder_b, exist_ok=True)
    
    # 读取图像
    image = cv2.imread(input_image)
    if image is None:
        print(f"错误: 无法读取图像 {input_image}")
        return False
    
    # 获取文件名
    filename = os.path.basename(input_image)
    print(f"处理图像: {filename}")
    
    # 使用多阶段文字去除流程处理图像
    processed_image = remove_text_multi_stage(image, filename)
    
    # 检查处理后的图像是否仍然含有文字
    has_text_result = has_text(processed_image, filename)
    
    # 保存到适当的文件夹
    output_folder = output_folder_a if has_text_result else output_folder_b
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, processed_image)
    
    print(f"图像已处理并保存到: {output_path}")
    print(f"文字检测结果: {'含有文字' if has_text_result else '不含文字'}")
    
    return True

if __name__ == "__main__":
    # 设置路径
    input_image = "test_images/french_text/page_6.jpg"
    output_folder_a = "processed_images/folder_a"
    output_folder_b = "processed_images/folder_b"
    
    # 处理图像
    process_image(input_image, output_folder_a, output_folder_b)
