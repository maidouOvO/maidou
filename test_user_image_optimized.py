import os
import cv2
import numpy as np
from optimized_text_remover import OptimizedTextRemover

def resize_image(image, max_size=1024):
    """
    调整图像大小，保持纵横比
    
    Args:
        image: 输入图像
        max_size: 最大尺寸
        
    Returns:
        调整大小后的图像
    """
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        if h > w:
            new_h = max_size
            new_w = int(w * max_size / h)
        else:
            new_w = max_size
            new_h = int(h * max_size / w)
        return cv2.resize(image, (new_w, new_h))
    return image

def test_on_user_image():
    """
    测试优化的文本去除功能在用户提供的图像上的效果
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
    resized_image = resize_image(image, max_size=1024)
    
    # 初始化优化的文本去除器，使用更保守的参数
    remover = OptimizedTextRemover(
        padding=2,  # 减小填充大小以避免过度扩展
        inpaint_method=cv2.INPAINT_TELEA,
        min_area=150,  # 增加最小面积以过滤小噪点
        min_aspect_ratio=0.3,  # 调整纵横比范围以更好地匹配文本
        max_aspect_ratio=6,
        epsilon_factor=0.02,
        char_min_area=30,  # 增加最小字符面积
        char_max_area=800,  # 减小最大字符面积以避免误检测大型角色特征
        char_min_aspect_ratio=0.4,  # 调整字符纵横比
        char_max_aspect_ratio=3,
        char_min_solidity=0.5,  # 增加最小实心度以更好地匹配文本字符
        dilation_kernel_size=2,  # 减小膨胀核大小以避免过度扩展
        gradient_threshold=30,  # 增加梯度阈值以更好地区分文本和非文本
        connectivity_threshold=0.2  # 增加连通性阈值以更好地区分孤立文本和连接特征
    )
    
    # 去除文本
    try:
        print("开始处理用户图像...")
        result, expanded_mask, filtered_mask = remover.remove_text(resized_image)
        
        # 保存结果
        filename = os.path.basename(image_path)
        base_name, ext = os.path.splitext(filename)
        
        # 保存原始图像
        original_path = os.path.join(output_dir, f"{base_name}_original{ext}")
        cv2.imwrite(original_path, resized_image)
        
        # 保存去除文本后的图像
        result_path = os.path.join(output_dir, f"{base_name}_result{ext}")
        cv2.imwrite(result_path, result)
        
        # 保存扩展遮罩
        expanded_mask_path = os.path.join(output_dir, f"{base_name}_expanded_mask{ext}")
        cv2.imwrite(expanded_mask_path, expanded_mask)
        
        # 保存过滤后的遮罩
        filtered_mask_path = os.path.join(output_dir, f"{base_name}_filtered_mask{ext}")
        cv2.imwrite(filtered_mask_path, filtered_mask)
        
        print(f"用户图像 {filename} 处理完成")
        print(f"结果已保存到 {output_dir}")
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {e}")

if __name__ == "__main__":
    test_on_user_image()
