#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import cv2
import numpy as np
import torch
import time
from collections import OrderedDict
import torch.backends.cudnn as cudnn

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 导入CRAFT模型和工具
craft_dir = os.path.join(current_dir, 'CRAFT-pytorch')
if craft_dir not in sys.path:
    sys.path.append(craft_dir)

# 直接导入模块
sys.path.insert(0, os.path.join(current_dir, 'CRAFT-pytorch'))
from craft import CRAFT
from craft_utils import getDetBoxes, adjustResultCoordinates
from imgproc import resize_aspect_ratio, normalizeMeanVariance

def copyStateDict(state_dict):
    """复制模型状态字典"""
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def load_craft_model(weights_path, cuda=False):
    """
    加载CRAFT模型
    
    参数:
        weights_path: 权重文件路径
        cuda: 是否使用CUDA
        
    返回:
        net: 加载的模型
    """
    # 初始化模型
    net = CRAFT()
    
    print(f'正在加载预训练模型: {weights_path}')
    if cuda:
        net.load_state_dict(copyStateDict(torch.load(weights_path)))
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False
    else:
        net.load_state_dict(copyStateDict(torch.load(weights_path, map_location='cpu')))
    
    net.eval()
    return net

def detect_text_regions(net, image, text_threshold=0.7, link_threshold=0.4, low_text=0.4, 
                        cuda=False, poly=False, refine_net=None, show_time=False,
                        canvas_size=1280, mag_ratio=1.5):
    """
    检测图像中的文本区域
    
    参数:
        net: CRAFT模型
        image: 输入图像
        text_threshold: 文本置信度阈值
        link_threshold: 链接置信度阈值
        low_text: 低文本置信度阈值
        cuda: 是否使用CUDA
        poly: 是否使用多边形
        refine_net: 精细化网络
        show_time: 是否显示处理时间
        canvas_size: 画布大小
        mag_ratio: 放大比例
        
    返回:
        boxes: 检测到的文本框
        polys: 检测到的文本多边形
        ret_score_text: 文本得分图
    """
    t0 = time.time()
    
    # 调整图像大小
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(image, canvas_size, 
                                                                interpolation=cv2.INTER_LINEAR, 
                                                                mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio
    
    # 预处理图像
    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = x.unsqueeze(0)                          # [c, h, w] to [b, c, h, w]
    
    # 使用CUDA（如果可用）
    if cuda:
        x = x.cuda()
    
    # 前向传播
    with torch.no_grad():
        y, feature = net(x)
    
    # 获取输出
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()
    
    # 获取文本框
    boxes, labels, mapper = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)
    
    # 调整坐标
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = boxes
    
    # 过滤太小的文本框
    for k in range(len(polys)):
        if polys[k] is not None:
            if len(polys[k]) < 4:
                polys[k] = None
    
    t1 = time.time()
    
    if show_time:
        print(f"文本检测耗时: {t1 - t0:.3f}s")
    
    return boxes, polys, score_text, labels

def create_text_mask(image, polys, dilation_size=7):
    """
    创建文本掩码
    
    参数:
        image: 输入图像
        polys: 文本多边形
        dilation_size: 膨胀核大小
        
    返回:
        mask: 文本掩码
    """
    # 创建掩码
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # 在掩码上绘制文本区域
    for poly in polys:
        if poly is not None:
            # 将多边形转换为整数坐标
            poly_points = np.array(poly, dtype=np.int32)
            cv2.fillPoly(mask, [poly_points], 255)  # 填充多边形区域
    
    # 膨胀掩码以扩大文本区域
    kernel = np.ones((dilation_size, dilation_size), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    return mask

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
    
    # 检查权重文件是否存在
    if not os.path.exists(weights_path):
        print(f"错误: CRAFT模型权重文件不存在: {weights_path}")
        return
    
    # 创建输出目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取图像: {image_path}")
        return
    
    # 加载CRAFT模型
    net = load_craft_model(weights_path, cuda)
    
    # 测试不同参数组合
    test_params = [
        {
            'name': 'default',
            'text_threshold': 0.7,
            'link_threshold': 0.4,
            'low_text': 0.4,
            'dilation_size': 7
        },
        {
            'name': 'high_precision',
            'text_threshold': 0.8,
            'link_threshold': 0.5,
            'low_text': 0.5,
            'dilation_size': 5
        },
        {
            'name': 'high_recall',
            'text_threshold': 0.6,
            'link_threshold': 0.3,
            'low_text': 0.3,
            'dilation_size': 9
        }
    ]
    
    # 测试每组参数
    for params in test_params:
        print(f"测试参数组: {params['name']}")
        
        # 检测文本区域
        boxes, polys, score_text, _ = detect_text_regions(
            net, image,
            text_threshold=params['text_threshold'],
            link_threshold=params['link_threshold'],
            low_text=params['low_text'],
            cuda=cuda,
            show_time=True
        )
        
        # 创建文本掩码
        mask = create_text_mask(image, polys, dilation_size=params['dilation_size'])
        
        # 使用掩码移除文本
        result = cv2.inpaint(image, mask, 10, cv2.INPAINT_NS)
        
        # 保存结果
        if output_dir:
            # 保存检测结果
            result_path = os.path.join(output_dir, f"result_{params['name']}.jpg")
            mask_path = os.path.join(output_dir, f"mask_{params['name']}.jpg")
            
            # 在原图上绘制检测框
            img_draw = image.copy()
            for poly in polys:
                if poly is not None:
                    poly_points = np.array(poly, dtype=np.int32)
                    cv2.polylines(img_draw, [poly_points], True, (0, 0, 255), 2)
            
            # 保存结果
            cv2.imwrite(result_path, result)
            cv2.imwrite(mask_path, mask)
            cv2.imwrite(os.path.join(output_dir, f"boxes_{params['name']}.jpg"), img_draw)
            
            print(f"  结果保存到: {result_path}")
            print(f"  掩码保存到: {mask_path}")
        
        print(f"  检测到的文本框数量: {len([p for p in polys if p is not None])}")
        print()
    
    print("测试完成！")

if __name__ == "__main__":
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='测试CRAFT实现')
    parser.add_argument('--image', type=str, required=True, help='测试图像路径')
    parser.add_argument('--output', type=str, default=None, help='输出目录')
    parser.add_argument('--weights', type=str, default=None, help='CRAFT模型权重路径')
    parser.add_argument('--cuda', action='store_true', help='是否使用CUDA')
    
    args = parser.parse_args()
    
    # 测试CRAFT实现
    test_craft_implementation(args.image, args.output, args.weights, args.cuda)
