#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import urllib.request
import hashlib
import argparse

def download_file(url, output_path, expected_md5=None):
    """
    下载文件并验证MD5校验和
    
    参数:
        url: 下载URL
        output_path: 输出路径
        expected_md5: 预期的MD5校验和（可选）
    
    返回:
        success: 是否成功下载
    """
    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 如果文件已存在，检查MD5
    if os.path.exists(output_path):
        if expected_md5:
            # 计算文件的MD5
            with open(output_path, 'rb') as f:
                file_md5 = hashlib.md5(f.read()).hexdigest()
            
            # 如果MD5匹配，跳过下载
            if file_md5 == expected_md5:
                print(f"文件已存在且MD5校验和匹配: {output_path}")
                return True
            else:
                print(f"文件已存在但MD5校验和不匹配，重新下载: {output_path}")
        else:
            print(f"文件已存在，跳过下载: {output_path}")
            return True
    
    # 下载文件
    try:
        print(f"正在下载: {url} -> {output_path}")
        
        # 创建请求并添加User-Agent头
        req = urllib.request.Request(
            url, 
            data=None,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
            }
        )
        
        # 下载文件
        with urllib.request.urlopen(req) as response, open(output_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        
        # 如果指定了MD5，验证下载的文件
        if expected_md5:
            with open(output_path, 'rb') as f:
                file_md5 = hashlib.md5(f.read()).hexdigest()
            
            if file_md5 != expected_md5:
                print(f"警告: 下载的文件MD5校验和不匹配")
                print(f"  预期: {expected_md5}")
                print(f"  实际: {file_md5}")
                return False
        
        print(f"下载完成: {output_path}")
        return True
    
    except Exception as e:
        print(f"下载失败: {e}")
        return False

def download_craft_weights(output_dir=None, force=False):
    """
    下载CRAFT模型权重
    
    参数:
        output_dir: 输出目录
        force: 是否强制下载（即使文件已存在）
    
    返回:
        success: 是否成功下载
    """
    # 设置默认输出目录
    if output_dir is None:
        # 获取当前脚本所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(current_dir, 'weights')
    
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # CRAFT模型权重URL和MD5
    craft_url = "https://drive.google.com/uc?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ"
    craft_output_path = os.path.join(output_dir, 'craft_mlt_25k.pth')
    craft_md5 = "2f8227d2def4037cdb3b34389dcf9ec1"  # 预期的MD5校验和
    
    # 如果强制下载或文件不存在，删除现有文件
    if force and os.path.exists(craft_output_path):
        os.remove(craft_output_path)
    
    # 下载CRAFT模型权重
    success = download_file(craft_url, craft_output_path, craft_md5)
    
    return success

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='下载CRAFT模型权重')
    parser.add_argument('--output-dir', type=str, default=None, help='输出目录')
    parser.add_argument('--force', action='store_true', help='强制下载（即使文件已存在）')
    
    args = parser.parse_args()
    
    # 下载CRAFT模型权重
    success = download_craft_weights(args.output_dir, args.force)
    
    # 设置退出代码
    sys.exit(0 if success else 1)
