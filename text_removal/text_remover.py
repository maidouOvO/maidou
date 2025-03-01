import os
import numpy as np
import cv2
from .text_detector import TextDetector

class TextRemover:
    def __init__(self, 
                 detector=None,
                 padding=5,
                 inpaint_method=cv2.INPAINT_TELEA,
                 character_detection_threshold=127,
                 dilation_kernel_size=3,
                 min_contour_area=10):
        """
        初始化文字去除器
        
        Args:
            detector: 文本检测器实例，如果为None则创建默认检测器
            padding: 文本区域填充大小，用于创建扩展遮罩
            inpaint_method: 图像修复方法，可选cv2.INPAINT_TELEA或cv2.INPAINT_NS
            character_detection_threshold: 字符检测阈值，用于二值化
            dilation_kernel_size: 膨胀核大小，用于扩展遮罩
            min_contour_area: 最小轮廓面积，用于过滤噪声
        """
        self.detector = detector if detector else TextDetector()
        self.padding = padding
        self.inpaint_method = inpaint_method
        self.character_detection_threshold = character_detection_threshold
        self.dilation_kernel_size = dilation_kernel_size
        self.min_contour_area = min_contour_area
    
    def _create_expanded_mask(self, image, polys):
        """
        创建扩展遮罩，覆盖文本区域并添加一些填充
        
        Args:
            image: 输入图像
            polys: 文本多边形列表
            
        Returns:
            扩展遮罩
        """
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for poly in polys:
            if poly is not None:
                # 将多边形转换为整数坐标
                poly = np.array(poly, dtype=np.int32)
                # 绘制填充多边形
                cv2.fillPoly(mask, [poly], 255)
        
        # 应用膨胀操作以扩展遮罩
        if self.padding > 0:
            kernel = np.ones((self.padding, self.padding), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            
        return mask
    
    def _create_refined_mask(self, image, expanded_mask):
        """
        在扩展遮罩内创建精细遮罩，精确匹配文本形状
        
        Args:
            image: 输入图像
            expanded_mask: 扩展遮罩
            
        Returns:
            精细遮罩
        """
        # 转换为灰度图像
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 应用自适应阈值处理以检测文本形状
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 仅保留扩展遮罩内的区域
        refined_mask = cv2.bitwise_and(binary, expanded_mask)
        
        # 查找轮廓以进行单字符检测
        contours, _ = cv2.findContours(
            refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # 创建新的遮罩，仅包含符合条件的轮廓
        character_mask = np.zeros_like(refined_mask)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_contour_area:
                # 检查是否为文本特征而非角色特征
                if self._is_text_feature(contour, gray):
                    cv2.drawContours(character_mask, [contour], -1, 255, -1)
        
        # 应用小的膨胀以确保完全覆盖文本
        kernel = np.ones((self.dilation_kernel_size, self.dilation_kernel_size), np.uint8)
        character_mask = cv2.dilate(character_mask, kernel, iterations=1)
        
        return character_mask
    
    def _is_text_feature(self, contour, gray_image):
        """
        判断轮廓是否为文本特征而非角色特征
        
        Args:
            contour: 轮廓
            gray_image: 灰度图像
            
        Returns:
            布尔值，表示是否为文本特征
        """
        # 计算轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)
        
        # 计算轮廓的纵横比
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # 计算轮廓的面积与边界框面积的比率
        area = cv2.contourArea(contour)
        rect_area = w * h
        extent = float(area) / rect_area if rect_area > 0 else 0
        
        # 文本特征通常具有适中的纵横比和较高的extent
        is_text_by_shape = (0.1 < aspect_ratio < 10) and (extent > 0.3)
        
        # 检查轮廓是否与其他非文本特征连接
        # 这需要分析轮廓周围的区域
        is_isolated = self._is_isolated_feature(contour, gray_image)
        
        # 检查轮廓大小是否合适
        is_appropriate_size = (10 < w < 100) and (10 < h < 100)
        
        return is_text_by_shape and is_isolated and is_appropriate_size
    
    def _is_isolated_feature(self, contour, gray_image):
        """
        判断轮廓是否为孤立特征（文本通常是孤立的）
        
        Args:
            contour: 轮廓
            gray_image: 灰度图像
            
        Returns:
            布尔值，表示是否为孤立特征
        """
        # 获取轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)
        
        # 扩展边界框以检查周围区域
        padding = 10
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(gray_image.shape[1], x + w + padding)
        y2 = min(gray_image.shape[0], y + h + padding)
        
        # 创建轮廓遮罩
        mask = np.zeros_like(gray_image)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # 获取扩展区域
        region = gray_image[y1:y2, x1:x2].copy()
        region_mask = mask[y1:y2, x1:x2].copy()
        
        # 计算轮廓外部区域的平均梯度
        # 文本通常与背景有明显的边界，而角色特征可能与其他特征连续
        gradient_x = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # 创建轮廓边界遮罩
        boundary_mask = cv2.dilate(region_mask, np.ones((3, 3), np.uint8)) - region_mask
        
        # 如果边界遮罩中没有像素，则认为是孤立的
        if np.sum(boundary_mask) == 0:
            return True
        
        # 计算边界处的平均梯度
        boundary_gradient = np.mean(gradient_magnitude[boundary_mask > 0])
        
        # 高梯度表示明显的边界，可能是文本
        return boundary_gradient > 30
    
    def remove_text(self, image):
        """
        从图像中去除文本
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            去除文本后的图像，文本区域遮罩，精细遮罩
        """
        # 检测文本区域
        _, polys, _ = self.detector.detect(image)
        
        # 创建扩展遮罩
        expanded_mask = self._create_expanded_mask(image, polys)
        
        # 创建精细遮罩
        refined_mask = self._create_refined_mask(image, expanded_mask)
        
        # 使用精细遮罩进行图像修复
        result = cv2.inpaint(image, refined_mask, 3, self.inpaint_method)
        
        return result, expanded_mask, refined_mask
