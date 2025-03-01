import os
import argparse
import cv2
import numpy as np
from text_removal.text_detector import TextDetector
from text_removal.text_remover import TextRemover

def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='文本检测与去除演示')
    parser.add_argument('--image_path', type=str, required=True, help='图像路径')
    parser.add_argument('--output_dir', type=str, default='results', help='输出目录')
    parser.add_argument('--cuda', action='store_true', help='使用CUDA进行推理')
    parser.add_argument('--text_threshold', type=float, default=0.7, help='文本置信度阈值')
    parser.add_argument('--low_text', type=float, default=0.4, help='文本低边界分数')
    parser.add_argument('--link_threshold', type=float, default=0.4, help='链接置信度阈值')
    parser.add_argument('--padding', type=int, default=5, help='文本区域填充大小')
    parser.add_argument('--inpaint_method', type=str, default='telea', 
                        choices=['telea', 'ns'], help='图像修复方法')
    parser.add_argument('--char_threshold', type=int, default=127, help='字符检测阈值')
    parser.add_argument('--dilation_size', type=int, default=3, help='膨胀核大小')
    parser.add_argument('--min_contour_area', type=int, default=10, help='最小轮廓面积')
    
    return parser.parse_args()

def main():
    """
    主函数
    """
    # 解析参数
    args = parse_arguments()
    
    # 确保输出目录存在
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 加载图像
    if not os.path.isfile(args.image_path):
        print(f"找不到图像: {args.image_path}")
        return
    
    image = cv2.imread(args.image_path)
    if image is None:
        print(f"无法读取图像: {args.image_path}")
        return
    
    # 初始化文本检测器
    detector = TextDetector(
        text_threshold=args.text_threshold,
        low_text=args.low_text,
        link_threshold=args.link_threshold,
        cuda=args.cuda
    )
    
    # 初始化文本去除器
    inpaint_method = cv2.INPAINT_TELEA if args.inpaint_method == 'telea' else cv2.INPAINT_NS
    remover = TextRemover(
        detector=detector,
        padding=args.padding,
        inpaint_method=inpaint_method,
        character_detection_threshold=args.char_threshold,
        dilation_kernel_size=args.dilation_size,
        min_contour_area=args.min_contour_area
    )
    
    # 去除文本
    try:
        result, expanded_mask, refined_mask = remover.remove_text(image)
        
        # 保存结果
        filename = os.path.basename(args.image_path)
        base_name, ext = os.path.splitext(filename)
        
        # 保存原始图像
        original_path = os.path.join(args.output_dir, f"{base_name}_original{ext}")
        cv2.imwrite(original_path, image)
        
        # 保存去除文本后的图像
        result_path = os.path.join(args.output_dir, f"{base_name}_result{ext}")
        cv2.imwrite(result_path, result)
        
        # 保存扩展遮罩
        expanded_mask_path = os.path.join(args.output_dir, f"{base_name}_expanded_mask{ext}")
        cv2.imwrite(expanded_mask_path, expanded_mask)
        
        # 保存精细遮罩
        refined_mask_path = os.path.join(args.output_dir, f"{base_name}_refined_mask{ext}")
        cv2.imwrite(refined_mask_path, refined_mask)
        
        print(f"文本去除完成")
        print(f"结果已保存到 {args.output_dir}")
    except Exception as e:
        print(f"文本去除过程中出现错误: {e}")

if __name__ == "__main__":
    main()
