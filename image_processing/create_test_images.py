#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np

def create_test_images():
    # 确保文件夹存在
    folder1 = os.path.expanduser("~/image_processing_project/folder1")
    os.makedirs(folder1, exist_ok=True)
    
    # 创建不同类型的测试图片
    
    # 1. 纯色背景带文字
    img1 = Image.new('RGB', (800, 600), color=(255, 255, 255))
    draw1 = ImageDraw.Draw(img1)
    try:
        # 尝试加载系统字体
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
    except IOError:
        # 如果找不到指定字体，使用默认字体
        font = ImageFont.load_default()
    
    draw1.text((200, 250), "这是测试文字 Test Text", fill=(0, 0, 0), font=font)
    img1.save(os.path.join(folder1, "test_image1.png"))
    
    # 2. 渐变背景带文字
    img2 = Image.new('RGB', (800, 600))
    draw2 = ImageDraw.Draw(img2)
    
    # 创建渐变背景
    for y in range(600):
        for x in range(800):
            r = int(255 * x / 800)
            g = int(255 * y / 600)
            b = 100
            draw2.point((x, y), fill=(r, g, b))
    
    draw2.text((200, 250), "渐变背景文字 Gradient Text", fill=(255, 255, 255), font=font)
    img2.save(os.path.join(folder1, "test_image2.png"))
    
    # 3. 模拟照片带文字
    img3 = Image.new('RGB', (800, 600))
    draw3 = ImageDraw.Draw(img3)
    
    # 创建随机噪点背景模拟照片
    np_array = np.random.randint(100, 200, (600, 800, 3), dtype=np.uint8)
    img3 = Image.fromarray(np_array)
    draw3 = ImageDraw.Draw(img3)
    
    draw3.text((150, 250), "照片上的文字 Photo Text", fill=(0, 0, 0), font=font)
    draw3.text((150, 300), "第二行文字 Second Line", fill=(0, 0, 0), font=font)
    img3.save(os.path.join(folder1, "test_image3.png"))
    
    # 4. 无文字图片
    img4 = Image.new('RGB', (800, 600), color=(200, 200, 200))
    img4.save(os.path.join(folder1, "test_image4_no_text.png"))
    
    print(f"已创建测试图片，保存在 {folder1}")

if __name__ == "__main__":
    create_test_images()
