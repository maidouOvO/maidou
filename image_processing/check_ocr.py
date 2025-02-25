#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytesseract
from PIL import Image
import os

def check_ocr_results():
    """检查原始图片中的OCR文字识别结果"""
    folder1 = os.path.expanduser('~/image_processing_project/folder1')
    
    print("原始图片OCR识别结果:")
    print("-" * 50)
    
    for file in os.listdir(folder1):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder1, file)
            img = Image.open(img_path)
            
            # 使用不同的配置尝试OCR识别
            text_default = pytesseract.image_to_string(img, lang='chi_sim+eng')
            text_config1 = pytesseract.image_to_string(img, lang='chi_sim+eng', config='--psm 6')
            text_config2 = pytesseract.image_to_string(img, lang='chi_sim+eng', config='--psm 11')
            
            print(f'文件: {file}')
            print(f'默认配置识别文字: {text_default.strip() or "[无文字]"}')
            print(f'配置1识别文字: {text_config1.strip() or "[无文字]"}')
            print(f'配置2识别文字: {text_config2.strip() or "[无文字]"}')
            print("-" * 50)

if __name__ == "__main__":
    check_ocr_results()
