import os
import cv2
import numpy as np
from ocr_text_remover import OCRTextRemover

def test_on_user_image():
    """
    测试OCR文本去除功能在用户提供的图像上的效果
    """
    # 设置参数
    image_path = 'data/test1.jpg'
    output_dir = 'results'
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 加载图像
    if not os.path.isfile(image_path):
        print(f"找不到图像: {image_path}")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    
    # 调整图像大小以加快处理速度
    h, w = image.shape[:2]
    if max(h, w) > 1024:
        if h > w:
            new_h = 1024
            new_w = int(w * 1024 / h)
        else:
            new_w = 1024
            new_h = int(h * 1024 / w)
        image = cv2.resize(image, (new_w, new_h))
    
    # 初始化OCR文本去除器，针对用户图像调整参数
    remover = OCRTextRemover(
        text_threshold=0.5,  # 降低文本阈值以检测更多文本
        link_threshold=0.3,  # 降低链接阈值以连接更多文本区域
        low_text=0.3,        # 降低低文本阈值以检测更多文本
        cuda=False,
        padding=5,           # 增加填充大小以确保覆盖整个文本区域
        inpaint_method=cv2.INPAINT_TELEA,
        min_area=50,
        min_aspect_ratio=0.1,
        max_aspect_ratio=10,
        epsilon_factor=0.02,
        char_min_area=20,
        char_max_area=5000,  # 增大最大字符面积以处理大字体
        char_min_aspect_ratio=0.1,
        char_max_aspect_ratio=10,
        char_min_solidity=0.2,
        dilation_kernel_size=5  # 增大膨胀核大小以确保覆盖整个文本
    )
    
    # 去除文本
    try:
        print("开始处理用户图像...")
        result, region_mask, filtered_mask = remover.remove_text(image)
        
        # 保存结果
        filename = os.path.basename(image_path)
        base_name, ext = os.path.splitext(filename)
        
        # 保存原始图像
        original_path = os.path.join(output_dir, f"{base_name}_ocr_original{ext}")
        cv2.imwrite(original_path, image)
        
        # 保存去除文本后的图像
        result_path = os.path.join(output_dir, f"{base_name}_ocr_result{ext}")
        cv2.imwrite(result_path, result)
        
        # 保存文本区域遮罩
        region_mask_path = os.path.join(output_dir, f"{base_name}_ocr_region_mask{ext}")
        cv2.imwrite(region_mask_path, region_mask)
        
        # 保存文本形状遮罩
        filtered_mask_path = os.path.join(output_dir, f"{base_name}_ocr_filtered_mask{ext}")
        cv2.imwrite(filtered_mask_path, filtered_mask)
        
        print(f"用户图像 {filename} 处理完成")
        print(f"结果已保存到 {output_dir}")
        
        # 显示处理前后的对比
        print(f"原始图像: {original_path}")
        print(f"处理后图像: {result_path}")
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {e}")

def test_with_different_parameters():
    """
    使用不同的参数测试OCR文本去除功能
    """
    # 设置参数
    image_path = 'data/test1.jpg'
    output_dir = 'results'
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 加载图像
    if not os.path.isfile(image_path):
        print(f"找不到图像: {image_path}")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    
    # 调整图像大小以加快处理速度
    h, w = image.shape[:2]
    if max(h, w) > 1024:
        if h > w:
            new_h = 1024
            new_w = int(w * 1024 / h)
        else:
            new_w = 1024
            new_h = int(h * 1024 / w)
        image = cv2.resize(image, (new_w, new_h))
    
    # 定义不同的参数组合
    parameter_sets = [
        {
            'name': 'aggressive',
            'params': {
                'text_threshold': 0.4,
                'link_threshold': 0.2,
                'low_text': 0.2,
                'padding': 7,
                'dilation_kernel_size': 7,
                'char_max_area': 8000
            }
        },
        {
            'name': 'balanced',
            'params': {
                'text_threshold': 0.5,
                'link_threshold': 0.3,
                'low_text': 0.3,
                'padding': 5,
                'dilation_kernel_size': 5,
                'char_max_area': 5000
            }
        },
        {
            'name': 'conservative',
            'params': {
                'text_threshold': 0.6,
                'link_threshold': 0.4,
                'low_text': 0.4,
                'padding': 3,
                'dilation_kernel_size': 3,
                'char_max_area': 3000
            }
        }
    ]
    
    # 测试每组参数
    for param_set in parameter_sets:
        name = param_set['name']
        params = param_set['params']
        
        print(f"使用 {name} 参数测试...")
        
        # 初始化OCR文本去除器
        remover = OCRTextRemover(
            text_threshold=params['text_threshold'],
            link_threshold=params['link_threshold'],
            low_text=params['low_text'],
            cuda=False,
            padding=params['padding'],
            inpaint_method=cv2.INPAINT_TELEA,
            min_area=50,
            min_aspect_ratio=0.1,
            max_aspect_ratio=10,
            epsilon_factor=0.02,
            char_min_area=20,
            char_max_area=params['char_max_area'],
            char_min_aspect_ratio=0.1,
            char_max_aspect_ratio=10,
            char_min_solidity=0.2,
            dilation_kernel_size=params['dilation_kernel_size']
        )
        
        # 去除文本
        try:
            result, region_mask, filtered_mask = remover.remove_text(image)
            
            # 保存结果
            filename = os.path.basename(image_path)
            base_name, ext = os.path.splitext(filename)
            
            # 保存去除文本后的图像
            result_path = os.path.join(output_dir, f"{base_name}_ocr_{name}_result{ext}")
            cv2.imwrite(result_path, result)
            
            # 保存文本区域遮罩
            region_mask_path = os.path.join(output_dir, f"{base_name}_ocr_{name}_region_mask{ext}")
            cv2.imwrite(region_mask_path, region_mask)
            
            # 保存文本形状遮罩
            filtered_mask_path = os.path.join(output_dir, f"{base_name}_ocr_{name}_filtered_mask{ext}")
            cv2.imwrite(filtered_mask_path, filtered_mask)
            
            print(f"参数组 {name} 处理完成")
        except Exception as e:
            print(f"使用参数组 {name} 处理图像时出错: {e}")

if __name__ == "__main__":
    # 测试用户图像
    test_on_user_image()
    
    # 使用不同的参数测试
    test_with_different_parameters()
