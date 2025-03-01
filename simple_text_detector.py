import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

# 创建一个简单的CRAFT模型
class SimpleCRAFT(nn.Module):
    def __init__(self):
        super(SimpleCRAFT, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(512, 2, kernel_size=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

# 文本检测器类
class SimpleTextDetector:
    def __init__(self):
        self.model = SimpleCRAFT()
        
    def detect(self, image):
        """
        检测图像中的文本区域
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            boxes: 文本边界框
            mask: 文本区域遮罩
        """
        # 转换为灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用自适应阈值处理检测文本
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 应用形态学操作以改善文本区域
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)
        binary = cv2.erode(binary, kernel, iterations=1)
        
        # 查找轮廓
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # 过滤轮廓
        boxes = []
        for contour in contours:
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            if area < 50:  # 过滤小轮廓
                continue
                
            # 获取边界框
            x, y, w, h = cv2.boundingRect(contour)
            
            # 计算纵横比
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # 文本通常具有适中的纵横比
            if 0.1 < aspect_ratio < 10:
                # 转换为CRAFT格式的框
                box = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32)
                boxes.append(box)
        
        # 创建文本区域遮罩
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for box in boxes:
            box = box.astype(np.int32)
            cv2.fillPoly(mask, [box], 255)
        
        return boxes, mask

# 测试函数
def test_text_detection():
    # 设置参数
    image_path = 'data/sample.jpg'
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
    
    # 初始化文本检测器
    detector = SimpleTextDetector()
    
    # 检测文本区域
    try:
        boxes, mask = detector.detect(image)
        
        # 绘制检测结果
        result_img = image.copy()
        
        # 绘制边界框
        for box in boxes:
            box = box.astype(np.int32)
            cv2.polylines(result_img, [box], True, color=(0, 0, 255), thickness=2)
        
        # 保存结果
        filename = os.path.basename(image_path)
        base_name, ext = os.path.splitext(filename)
        
        # 保存检测结果图像
        result_path = os.path.join(output_dir, f"{base_name}_result{ext}")
        cv2.imwrite(result_path, result_img)
        
        # 保存遮罩
        mask_path = os.path.join(output_dir, f"{base_name}_mask{ext}")
        cv2.imwrite(mask_path, mask)
        
        print(f"检测到 {len(boxes)} 个文本区域")
        print(f"结果已保存到 {output_dir}")
    except Exception as e:
        print(f"文本检测过程中出现错误: {e}")

if __name__ == "__main__":
    test_text_detection()
