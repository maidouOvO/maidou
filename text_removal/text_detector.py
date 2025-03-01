import os
import numpy as np
import cv2
from collections import OrderedDict

class TextDetector:
    def __init__(self, 
                 trained_model='craft/weights/craft_mlt_25k.pth',
                 text_threshold=0.7,
                 low_text=0.4,
                 link_threshold=0.4,
                 cuda=False,
                 canvas_size=1280,
                 mag_ratio=1.5,
                 poly=False):
        """
        初始化文本检测器
        
        Args:
            trained_model: 预训练模型路径（不使用）
            text_threshold: 文本置信度阈值
            low_text: 文本低边界分数
            link_threshold: 链接置信度阈值
            cuda: 是否使用CUDA进行推理（不使用）
            canvas_size: 推理图像大小（不使用）
            mag_ratio: 图像放大比例（不使用）
            poly: 是否启用多边形类型（不使用）
        """
        self.text_threshold = text_threshold
        self.low_text = low_text
        self.link_threshold = link_threshold
        self.cuda = cuda
        self.canvas_size = canvas_size
        self.mag_ratio = mag_ratio
        self.poly = poly
        
        # 注意：我们使用简化版本的文本检测，不需要加载CRAFT模型
        print("使用简化版本的文本检测器，不加载CRAFT模型")
    
    def _copy_state_dict(self, state_dict):
        """
        复制模型状态字典
        """
        if list(state_dict.keys())[0].startswith("module"):
            start_idx = 1
        else:
            start_idx = 0
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = ".".join(k.split(".")[start_idx:])
            new_state_dict[name] = v
        return new_state_dict
    
    def detect(self, image):
        """
        检测图像中的文本区域，特别优化用于检测大字体文本
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            boxes: 文本边界框
            polys: 文本多边形
            score_text: 文本得分热图
        """
        h, w = image.shape[:2]
        
        # 创建边界框和多边形列表
        boxes = []
        polys = []
        
        # 转换为灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用多种阈值处理方法来增强文本检测能力
        
        # 1. 使用自适应阈值处理 - 适用于小字体文本
        binary1 = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 15, 5
        )
        
        # 2. 使用自适应阈值处理 - 使用更大的块大小，适用于大字体文本
        binary2 = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 25, 10
        )
        
        # 3. 使用Otsu阈值处理 - 适用于高对比度文本
        _, binary3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 4. 使用固定阈值处理 - 适用于某些特定场景
        _, binary4 = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        
        # 合并二值图像
        binary = cv2.bitwise_or(binary1, binary2)
        binary = cv2.bitwise_or(binary, binary3)
        binary = cv2.bitwise_or(binary, binary4)
        
        # 应用形态学操作以增强文本区域
        # 使用更大的核和更多的迭代次数来处理大字体文本
        kernel = np.ones((7, 7), np.uint8)  # 增大核大小
        binary = cv2.dilate(binary, kernel, iterations=3)  # 增加迭代次数
        binary = cv2.erode(binary, kernel, iterations=1)
        
        # 应用闭操作以填充大字体文本中的空洞
        close_kernel = np.ones((15, 15), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # 处理每个轮廓
        for contour in contours:
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            
            # 过滤掉太小的轮廓，但允许更大的轮廓通过（适用于大字体文本）
            if area < 50:  # 降低最小面积阈值
                continue
            
            # 获取轮廓的边界框
            x, y, w, h = cv2.boundingRect(contour)
            
            # 计算纵横比
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # 放宽纵横比限制，以适应更多形状的文本
            if aspect_ratio < 0.05 or aspect_ratio > 20:
                continue
            
            # 计算轮廓的实心度（轮廓面积与其边界矩形面积之比）
            rect_area = w * h
            solidity = float(area) / rect_area if rect_area > 0 else 0
            
            # 过滤掉实心度太低的轮廓（通常不是文本）
            if solidity < 0.1:
                continue
            
            # 创建边界框
            box = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32)
            boxes.append(box)
            
            # 进行多边形近似
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            approx = approx.reshape(-1, 2).astype(np.float32)
            
            # 添加多边形
            polys.append(approx)
        
        # 创建得分热图
        score_text = np.zeros((h, w), dtype=np.float32)
        
        # 在得分热图上绘制轮廓
        for poly in polys:
            if poly is not None and len(poly) > 0:
                cv2.fillPoly(score_text, [poly.astype(np.int32)], 1.0)
        
        return boxes, polys, score_text
