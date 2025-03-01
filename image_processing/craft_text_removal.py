#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import torch
import sys
from PIL import Image

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 导入CRAFT文本检测模块
from craft_text_detection import load_craft_model, detect_and_create_mask

class CraftTextRemover:
    """
    使用CRAFT模型进行文本检测和移除的类
    """
    def __init__(self, weights_path, cuda=False, canvas_size=1280, mag_ratio=1.5):
        """
        初始化文本移除器
        
        参数:
            weights_path: CRAFT模型权重路径
            cuda: 是否使用CUDA
            canvas_size: 画布大小
            mag_ratio: 放大比例
        """
        self.cuda = cuda
        self.canvas_size = canvas_size
        self.mag_ratio = mag_ratio
        
        # 加载CRAFT模型
        if os.path.exists(weights_path):
            self.craft_net = load_craft_model(weights_path, cuda)
            self.model_loaded = True
        else:
            print(f"警告: 找不到CRAFT模型权重文件: {weights_path}")
            self.model_loaded = False
    
    def remove_text(self, image, text_threshold=0.7, link_threshold=0.4, low_text=0.4,
                   dilation_size=7, use_char_level=True, preserve_character_features=True,
                   inpaint_method='ns', inpaint_radius=10):
        """
        从图像中移除文本
        
        参数:
            image: 输入图像
            text_threshold: 文本置信度阈值
            link_threshold: 链接置信度阈值
            low_text: 低文本置信度阈值
            dilation_size: 膨胀核大小
            use_char_level: 是否使用字符级别检测
            preserve_character_features: 是否保留角色特征
            inpaint_method: 修复方法 ('ns', 'telea', 'advanced')
            inpaint_radius: 修复半径
            
        返回:
            result: 移除文本后的图像
            mask: 文本掩码
        """
        if not self.model_loaded:
            print("错误: CRAFT模型未加载")
            return image, None
        
        # 检测文本并创建掩码
        mask, text_polys = detect_and_create_mask(
            image, self.craft_net,
            text_threshold=text_threshold,
            link_threshold=link_threshold,
            low_text=low_text,
            cuda=self.cuda,
            canvas_size=self.canvas_size,
            mag_ratio=self.mag_ratio,
            dilation_size=dilation_size,
            use_char_level=use_char_level,
            preserve_character_features=preserve_character_features
        )
        
        # 如果没有检测到文本，直接返回原图
        if np.sum(mask) == 0:
            return image, mask
        
        # 根据指定的方法进行修复
        if inpaint_method == 'ns':
            # Navier-Stokes方法
            result = cv2.inpaint(image, mask, inpaint_radius, cv2.INPAINT_NS)
        elif inpaint_method == 'telea':
            # Telea方法
            result = cv2.inpaint(image, mask, inpaint_radius, cv2.INPAINT_TELEA)
        elif inpaint_method == 'advanced':
            # 高级修复方法（使用多种方法并选择最佳结果）
            from advanced_inpainting import advanced_inpaint
            result = advanced_inpaint(image, mask)
        else:
            # 默认使用Navier-Stokes方法
            result = cv2.inpaint(image, mask, inpaint_radius, cv2.INPAINT_NS)
        
        return result, mask
    
    def process_image(self, image_path, output_path=None, show_mask=False, **kwargs):
        """
        处理单个图像
        
        参数:
            image_path: 输入图像路径
            output_path: 输出图像路径
            show_mask: 是否显示掩码
            **kwargs: 传递给remove_text的参数
            
        返回:
            result: 移除文本后的图像
            mask: 文本掩码
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"错误: 无法读取图像: {image_path}")
            return None, None
        
        # 移除文本
        result, mask = self.remove_text(image, **kwargs)
        
        # 如果指定了输出路径，保存结果
        if output_path:
            # 创建输出目录（如果不存在）
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # 保存结果
            cv2.imwrite(output_path, result)
            
            # 如果需要显示掩码，保存掩码
            if show_mask and mask is not None:
                mask_path = os.path.splitext(output_path)[0] + "_mask.png"
                cv2.imwrite(mask_path, mask)
        
        return result, mask
    
    def batch_process(self, input_dir, output_dir, recursive=False, **kwargs):
        """
        批量处理图像
        
        参数:
            input_dir: 输入目录
            output_dir: 输出目录
            recursive: 是否递归处理子目录
            **kwargs: 传递给process_image的参数
            
        返回:
            processed_count: 处理的图像数量
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有图像文件
        image_files = []
        if recursive:
            for root, _, files in os.walk(input_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
                        image_files.append(os.path.join(root, file))
        else:
            image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))]
        
        # 处理每个图像
        processed_count = 0
        for image_path in image_files:
            # 计算输出路径
            rel_path = os.path.relpath(image_path, input_dir)
            output_path = os.path.join(output_dir, rel_path)
            
            # 处理图像
            print(f"处理图像: {rel_path}")
            _, mask = self.process_image(image_path, output_path, **kwargs)
            
            # 如果成功处理，增加计数
            if mask is not None:
                processed_count += 1
        
        return processed_count

def test_craft_text_removal(weights_path, image_path, output_dir=None, cuda=False):
    """
    测试CRAFT文本移除
    
    参数:
        weights_path: CRAFT模型权重路径
        image_path: 测试图像路径
        output_dir: 输出目录
        cuda: 是否使用CUDA
    """
    # 创建文本移除器
    remover = CraftTextRemover(weights_path, cuda)
    
    # 设置输出路径
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(image_path).split('.')[0]
        output_path = os.path.join(output_dir, f"{base_name}_removed.png")
    else:
        output_path = None
    
    # 处理图像
    result, mask = remover.process_image(
        image_path, output_path, show_mask=True,
        text_threshold=0.7, link_threshold=0.4, low_text=0.4,
        dilation_size=7, use_char_level=True, preserve_character_features=True,
        inpaint_method='advanced', inpaint_radius=10
    )
    
    # 如果没有指定输出目录，显示结果
    if output_dir is None and result is not None:
        # 转换为PIL图像并显示
        result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        result_pil.show()
        
        if mask is not None:
            mask_pil = Image.fromarray(mask)
            mask_pil.show()
    
    return result, mask

if __name__ == "__main__":
    # 测试代码
    import argparse
    
    parser = argparse.ArgumentParser(description='CRAFT文本移除')
    parser.add_argument('--weights', type=str, default='weights/craft_mlt_25k.pth', help='CRAFT模型权重路径')
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--output', type=str, default=None, help='输出目录')
    parser.add_argument('--cuda', action='store_true', help='是否使用CUDA')
    
    args = parser.parse_args()
    
    test_craft_text_removal(args.weights, args.image, args.output, args.cuda)
