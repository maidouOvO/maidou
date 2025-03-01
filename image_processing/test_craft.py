#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import cv2
import numpy as np
import argparse
from PIL import Image

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 导入CRAFT集成管道
from craft_integrated_pipeline import CraftIntegratedPipeline
from download_weights import download_craft_weights

def test_craft_implementation(image_path, output_dir=None, weights_path=None, cuda=False):
    """
    测试CRAFT实现
    
    参数:
        image_path: 测试图像路径
        output_dir: 输出目录
        weights_path: CRAFT模型权重路径
        cuda: 是否使用CUDA
    """
    # 如果未指定权重路径，使用默认路径
    if weights_path is None:
        weights_path = os.path.join(current_dir, 'weights', 'craft_mlt_25k.pth')
    
    # 检查权重文件是否存在，如果不存在则下载
    if not os.path.exists(weights_path):
        print(f"CRAFT模型权重文件不存在: {weights_path}")
        print("正在下载CRAFT模型权重...")
        weights_dir = os.path.dirname(weights_path)
        success = download_craft_weights(weights_dir)
        if not success:
            print("警告: 无法下载CRAFT模型权重，将使用默认参数")
    
    # 创建输出目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 创建集成管道
    pipeline = CraftIntegratedPipeline(weights_path, cuda)
    
    # 测试不同参数组合
    test_params = [
        {
            'name': 'default',
            'text_threshold': 0.7,
            'link_threshold': 0.4,
            'low_text': 0.4,
            'dilation_size': 7,
            'use_char_level': True,
            'preserve_character_features': True,
            'inpaint_method': 'advanced',
            'inpaint_radius': 10
        },
        {
            'name': 'high_precision',
            'text_threshold': 0.8,
            'link_threshold': 0.5,
            'low_text': 0.5,
            'dilation_size': 5,
            'use_char_level': True,
            'preserve_character_features': True,
            'inpaint_method': 'advanced',
            'inpaint_radius': 8
        },
        {
            'name': 'high_recall',
            'text_threshold': 0.6,
            'link_threshold': 0.3,
            'low_text': 0.3,
            'dilation_size': 9,
            'use_char_level': True,
            'preserve_character_features': True,
            'inpaint_method': 'advanced',
            'inpaint_radius': 12
        },
        {
            'name': 'no_char_level',
            'text_threshold': 0.7,
            'link_threshold': 0.4,
            'low_text': 0.4,
            'dilation_size': 7,
            'use_char_level': False,
            'preserve_character_features': True,
            'inpaint_method': 'advanced',
            'inpaint_radius': 10
        },
        {
            'name': 'no_preserve_character',
            'text_threshold': 0.7,
            'link_threshold': 0.4,
            'low_text': 0.4,
            'dilation_size': 7,
            'use_char_level': True,
            'preserve_character_features': False,
            'inpaint_method': 'advanced',
            'inpaint_radius': 10
        }
    ]
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取图像: {image_path}")
        return
    
    # 测试每组参数
    for params in test_params:
        print(f"测试参数组: {params['name']}")
        
        # 设置输出路径
        if output_dir:
            output_path = os.path.join(output_dir, f"result_{params['name']}.png")
            mask_path = os.path.join(output_dir, f"mask_{params['name']}.png")
        else:
            output_path = None
            mask_path = None
        
        # 处理图像
        result, mask, has_text = pipeline.process_image(
            image_path,
            output_path,
            text_threshold=params['text_threshold'],
            link_threshold=params['link_threshold'],
            low_text=params['low_text'],
            dilation_size=params['dilation_size'],
            use_char_level=params['use_char_level'],
            preserve_character_features=params['preserve_character_features'],
            inpaint_method=params['inpaint_method'],
            inpaint_radius=params['inpaint_radius']
        )
        
        # 保存掩码
        if output_dir and mask is not None:
            cv2.imwrite(mask_path, mask)
        
        # 显示结果
        print(f"  检测到文本: {has_text}")
        if output_dir:
            print(f"  结果保存到: {output_path}")
            print(f"  掩码保存到: {mask_path}")
        print()
    
    print("测试完成！")

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='测试CRAFT实现')
    parser.add_argument('--image', type=str, required=True, help='测试图像路径')
    parser.add_argument('--output', type=str, default=None, help='输出目录')
    parser.add_argument('--weights', type=str, default=None, help='CRAFT模型权重路径')
    parser.add_argument('--cuda', action='store_true', help='是否使用CUDA')
    
    args = parser.parse_args()
    
    # 测试CRAFT实现
    test_craft_implementation(args.image, args.output, args.weights, args.cuda)

if __name__ == "__main__":
    main()
