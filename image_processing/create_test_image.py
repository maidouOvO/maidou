#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os

def create_test_image(output_path, width=800, height=600, bg_color=(255, 255, 255)):
    """
    创建测试图像，包含不同大小和样式的文本
    
    参数:
        output_path: 输出图像路径
        width: 图像宽度
        height: 图像高度
        bg_color: 背景颜色
    """
    # 创建背景图像
    image = np.ones((height, width, 3), dtype=np.uint8) * np.array(bg_color, dtype=np.uint8)
    
    # 添加渐变背景
    for y in range(height):
        for x in range(width):
            # 创建渐变效果
            r = int(255 * (1 - y / height))
            g = int(255 * (1 - x / width))
            b = int(255 * (0.5 + 0.5 * (x + y) / (width + height)))
            image[y, x] = [b, g, r]
    
    # 添加一些形状（模拟角色特征）
    # 圆形（眼睛）
    cv2.circle(image, (200, 150), 30, (0, 0, 255), -1)
    cv2.circle(image, (300, 150), 30, (0, 0, 255), -1)
    
    # 椭圆（嘴巴）
    cv2.ellipse(image, (250, 250), (60, 30), 0, 0, 360, (0, 0, 0), -1)
    
    # 添加不同大小的文本
    # 小文本
    cv2.putText(image, "Small Text", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # 中等文本
    cv2.putText(image, "Medium Text", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    
    # 大文本（可能出现空心问题）
    cv2.putText(image, "Large Text", (50, 500), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 0), 5)
    
    # 添加一些作为角色特征的文本（不应被检测为要移除的文本）
    cv2.putText(image, "123", (190, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 添加一些复杂背景
    for i in range(20):
        x1 = np.random.randint(400, width - 100)
        y1 = np.random.randint(50, height - 100)
        x2 = x1 + np.random.randint(50, 100)
        y2 = y1 + np.random.randint(50, 100)
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
    
    # 在复杂背景上添加文本
    cv2.putText(image, "Text on Complex BG", (450, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # 保存图像
    cv2.imwrite(output_path, image)
    print(f"测试图像已创建: {output_path}")

if __name__ == "__main__":
    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_images")
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建测试图像
    output_path = os.path.join(output_dir, "test_image.jpg")
    create_test_image(output_path)
