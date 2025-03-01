#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import torch
import sys
from PIL import Image
import argparse
import time
import shutil

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 导入CRAFT文本检测和移除模块
from craft_text_detection import load_craft_model, detect_and_create_mask
from craft_text_removal import CraftTextRemover
from download_weights import download_craft_weights

class CraftIntegratedPipeline:
    """
    集成CRAFT文本检测和移除的图像处理管道
    """
    def __init__(self, weights_path=None, cuda=False, canvas_size=1280, mag_ratio=1.5):
        """
        初始化集成管道
        
        参数:
            weights_path: CRAFT模型权重路径
            cuda: 是否使用CUDA
            canvas_size: 画布大小
            mag_ratio: 放大比例
        """
        self.cuda = cuda
        self.canvas_size = canvas_size
        self.mag_ratio = mag_ratio
        
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
        
        # 创建文本移除器
        self.text_remover = CraftTextRemover(weights_path, cuda, canvas_size, mag_ratio)
    
    def process_image(self, image_path, output_path=None, show_mask=False,
                     text_threshold=0.7, link_threshold=0.4, low_text=0.4,
                     dilation_size=7, use_char_level=True, preserve_character_features=True,
                     inpaint_method='advanced', inpaint_radius=10):
        """
        处理单个图像
        
        参数:
            image_path: 输入图像路径
            output_path: 输出图像路径
            show_mask: 是否显示掩码
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
            has_text: 是否包含文本
        """
        # 处理图像
        result, mask = self.text_remover.process_image(
            image_path, output_path, show_mask,
            text_threshold=text_threshold,
            link_threshold=link_threshold,
            low_text=low_text,
            dilation_size=dilation_size,
            use_char_level=use_char_level,
            preserve_character_features=preserve_character_features,
            inpaint_method=inpaint_method,
            inpaint_radius=inpaint_radius
        )
        
        # 判断是否包含文本
        has_text = False
        if mask is not None and np.sum(mask) > 0:
            has_text = True
        
        return result, mask, has_text
    
    def batch_process(self, input_dir, output_dir_a, output_dir_b, recursive=False, **kwargs):
        """
        批量处理图像并根据是否包含文本分类到不同文件夹
        
        参数:
            input_dir: 输入目录
            output_dir_a: 包含文本的图像输出目录
            output_dir_b: 不包含文本的图像输出目录
            recursive: 是否递归处理子目录
            **kwargs: 传递给process_image的参数
            
        返回:
            result_stats: 处理结果统计
        """
        # 确保输出目录存在
        os.makedirs(output_dir_a, exist_ok=True)
        os.makedirs(output_dir_b, exist_ok=True)
        
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
        
        # 处理结果统计
        result_stats = {
            'total': len(image_files),
            'with_text': 0,
            'without_text': 0,
            'failed': 0,
            'processing_time': 0
        }
        
        # 处理每个图像
        start_time = time.time()
        for image_path in image_files:
            # 计算相对路径
            rel_path = os.path.relpath(image_path, input_dir)
            
            # 处理图像
            print(f"处理图像: {rel_path}")
            try:
                # 根据是否包含文本选择输出目录
                output_dir = output_dir_a  # 默认假设包含文本
                
                # 处理图像
                output_path = os.path.join(output_dir, rel_path)
                _, _, has_text = self.process_image(image_path, output_path, **kwargs)
                
                # 如果不包含文本，移动到B目录
                if not has_text:
                    output_dir = output_dir_b
                    new_output_path = os.path.join(output_dir, rel_path)
                    
                    # 确保输出目录存在
                    os.makedirs(os.path.dirname(new_output_path), exist_ok=True)
                    
                    # 移动文件
                    if os.path.exists(output_path):
                        shutil.move(output_path, new_output_path)
                    else:
                        # 如果文件不存在，直接复制原图
                        shutil.copy2(image_path, new_output_path)
                
                # 更新统计信息
                if has_text:
                    result_stats['with_text'] += 1
                else:
                    result_stats['without_text'] += 1
            
            except Exception as e:
                print(f"处理图像失败: {rel_path}, 错误: {e}")
                result_stats['failed'] += 1
        
        # 计算处理时间
        result_stats['processing_time'] = float(time.time() - start_time)
        
        return result_stats
    
    def generate_report(self, result_stats, output_dir):
        """
        生成处理报告
        
        参数:
            result_stats: 处理结果统计
            output_dir: 输出目录
        """
        # 创建报告文件
        report_path = os.path.join(output_dir, 'processing_report.md')
        
        # 写入报告
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 图像处理报告\n\n")
            
            f.write("## 处理统计\n")
            f.write(f"- 总处理图片数: {result_stats['total']}\n")
            f.write(f"- 含文字图片数: {result_stats['with_text']}\n")
            f.write(f"- 不含文字图片数: {result_stats['without_text']}\n")
            f.write(f"- 处理失败图片数: {result_stats['failed']}\n")
            f.write(f"- 总处理时间: {result_stats['processing_time']:.2f}秒\n\n")
            
            f.write("## 处理参数\n")
            f.write(f"- 画布大小: {self.canvas_size}\n")
            f.write(f"- 放大比例: {self.mag_ratio}\n")
            f.write(f"- 使用CUDA: {self.cuda}\n\n")
            
            f.write("## 注意事项\n")
            f.write("- 文件夹A中包含检测到文字的图片\n")
            f.write("- 文件夹B中包含未检测到文字的图片\n")
        
        print(f"处理报告已生成: {report_path}")

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='CRAFT集成图像处理管道')
    parser.add_argument('--input', type=str, required=True, help='输入目录')
    parser.add_argument('--output', type=str, required=True, help='输出目录')
    parser.add_argument('--weights', type=str, default=None, help='CRAFT模型权重路径')
    parser.add_argument('--cuda', action='store_true', help='是否使用CUDA')
    parser.add_argument('--recursive', action='store_true', help='是否递归处理子目录')
    parser.add_argument('--canvas-size', type=int, default=1280, help='画布大小')
    parser.add_argument('--mag-ratio', type=float, default=1.5, help='放大比例')
    parser.add_argument('--text-threshold', type=float, default=0.7, help='文本置信度阈值')
    parser.add_argument('--link-threshold', type=float, default=0.4, help='链接置信度阈值')
    parser.add_argument('--low-text', type=float, default=0.4, help='低文本置信度阈值')
    parser.add_argument('--dilation-size', type=int, default=7, help='膨胀核大小')
    parser.add_argument('--no-char-level', action='store_true', help='不使用字符级别检测')
    parser.add_argument('--no-preserve-character', action='store_true', help='不保留角色特征')
    parser.add_argument('--inpaint-method', type=str, default='advanced', 
                        choices=['ns', 'telea', 'advanced'], help='修复方法')
    parser.add_argument('--inpaint-radius', type=int, default=10, help='修复半径')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir_a = os.path.join(args.output, 'A_with_text')
    output_dir_b = os.path.join(args.output, 'B_without_text')
    os.makedirs(output_dir_a, exist_ok=True)
    os.makedirs(output_dir_b, exist_ok=True)
    
    # 创建集成管道
    pipeline = CraftIntegratedPipeline(
        weights_path=args.weights,
        cuda=args.cuda,
        canvas_size=args.canvas_size,
        mag_ratio=args.mag_ratio
    )
    
    # 批量处理图像
    result_stats = pipeline.batch_process(
        args.input,
        output_dir_a,
        output_dir_b,
        recursive=args.recursive,
        text_threshold=args.text_threshold,
        link_threshold=args.link_threshold,
        low_text=args.low_text,
        dilation_size=args.dilation_size,
        use_char_level=not args.no_char_level,
        preserve_character_features=not args.no_preserve_character,
        inpaint_method=args.inpaint_method,
        inpaint_radius=args.inpaint_radius
    )
    
    # 生成报告
    pipeline.generate_report(result_stats, args.output)
    
    print("处理完成！")
    print(f"总处理图片数: {result_stats['total']}")
    print(f"含文字图片数: {result_stats['with_text']}")
    print(f"不含文字图片数: {result_stats['without_text']}")
    print(f"处理失败图片数: {result_stats['failed']}")
    print(f"总处理时间: {result_stats['processing_time']:.2f}秒")

if __name__ == "__main__":
    main()
