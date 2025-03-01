import os
import cv2
import numpy as np
import argparse
from text_detector import TextDetector
from text_remover import TextRemover

def process_image(image_path, output_path, text_threshold=0.7, link_threshold=0.4, 
                low_text=0.4, cuda=False, expansion=5, block_size=11, c=2, 
                optimize_hollow=True):
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法加载图像: {image_path}")
        return
    
    # 初始化文本检测器和移除器
    detector = TextDetector(cuda=cuda)
    remover = TextRemover()
    
    # 检测文本区域
    boxes, polys = detector.detect(image, text_threshold, link_threshold, low_text)
    
    # 创建扩展遮罩
    expanded_mask = remover.create_expanded_mask(image, polys, expansion)
    
    # 精细化文字形状遮罩
    if optimize_hollow:
        refined_mask = remover.optimize_hollow_text_detection(image, expanded_mask)
    else:
        refined_mask = remover.refine_text_mask(image, expanded_mask, block_size, c)
    
    # 移除文字
    result = remover.remove_text(image, refined_mask)
    
    # 保存结果
    cv2.imwrite(output_path, result)
    
    # 保存中间结果用于调试
    debug_dir = os.path.dirname(output_path) + "/debug"
    os.makedirs(debug_dir, exist_ok=True)
    
    # 绘制检测框
    img_boxes = image.copy()
    for box in boxes:
        cv2.polylines(img_boxes, [np.array(box, np.int32).reshape((-1, 1, 2))], True, (0, 0, 255), 2)
    
    cv2.imwrite(f"{debug_dir}/boxes.jpg", img_boxes)
    cv2.imwrite(f"{debug_dir}/expanded_mask.jpg", expanded_mask)
    cv2.imwrite(f"{debug_dir}/refined_mask.jpg", refined_mask)
    
    print(f"处理完成，结果保存至: {output_path}")
    print(f"调试图像保存至: {debug_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='文本检测与移除')
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--output', type=str, required=True, help='输出图像路径')
    parser.add_argument('--text_threshold', type=float, default=0.7, help='文本检测阈值')
    parser.add_argument('--link_threshold', type=float, default=0.4, help='文本连接阈值')
    parser.add_argument('--low_text', type=float, default=0.4, help='低文本阈值')
    parser.add_argument('--cuda', action='store_true', help='使用CUDA加速')
    parser.add_argument('--expansion', type=int, default=5, help='遮罩扩展大小')
    parser.add_argument('--block_size', type=int, default=11, help='自适应阈值块大小')
    parser.add_argument('--c', type=int, default=2, help='自适应阈值常数')
    parser.add_argument('--optimize_hollow', action='store_true', default=True, help='优化中空字体检测')
    
    args = parser.parse_args()
    process_image(args.image, args.output, args.text_threshold, args.link_threshold, 
                args.low_text, args.cuda, args.expansion, args.block_size, args.c, args.optimize_hollow)
