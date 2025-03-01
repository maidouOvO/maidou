import os
import cv2
import numpy as np
import math

class PolygonTextDetector:
    def __init__(self, 
                 min_area=100, 
                 min_aspect_ratio=0.1, 
                 max_aspect_ratio=10,
                 epsilon_factor=0.02):
        """
        初始化多边形文本检测器
        
        Args:
            min_area: 最小轮廓面积
            min_aspect_ratio: 最小纵横比
            max_aspect_ratio: 最大纵横比
            epsilon_factor: 多边形近似的epsilon因子
        """
        self.min_area = min_area
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.epsilon_factor = epsilon_factor
    
    def detect(self, image):
        """
        检测图像中的文本多边形
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            多边形列表，文本区域遮罩
        """
        # 转换为灰度图像
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
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
        
        # 创建文本区域遮罩
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # 过滤轮廓并提取多边形
        polygons = []
        for contour in contours:
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            
            # 获取边界框
            x, y, w, h = cv2.boundingRect(contour)
            
            # 计算纵横比
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # 文本通常具有适中的纵横比
            if self.min_aspect_ratio < aspect_ratio < self.max_aspect_ratio:
                # 多边形近似
                epsilon = self.epsilon_factor * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # 如果多边形点数过多，可能不是文本
                if len(approx) <= 12:  # 通常文本字符的多边形点数不会太多
                    # 添加到多边形列表
                    polygons.append(approx)
                    
                    # 绘制到遮罩上
                    cv2.fillPoly(mask, [approx], 255)
        
        return polygons, mask
    
    def refine_polygons(self, image, polygons):
        """
        细化多边形，使其更好地匹配文本形状
        
        Args:
            image: 输入图像
            polygons: 初始多边形列表
            
        Returns:
            细化后的多边形列表，细化遮罩
        """
        # 转换为灰度图像
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 创建初始遮罩
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for poly in polygons:
            cv2.fillPoly(mask, [poly], 255)
        
        # 应用Canny边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 仅保留遮罩内的边缘
        edges = cv2.bitwise_and(edges, mask)
        
        # 应用形态学操作以连接边缘
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # 查找新的轮廓
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # 创建细化遮罩
        refined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # 过滤轮廓并提取细化多边形
        refined_polygons = []
        for contour in contours:
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            if area < self.min_area / 2:  # 使用较小的面积阈值
                continue
            
            # 多边形近似
            epsilon = self.epsilon_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 添加到细化多边形列表
            refined_polygons.append(approx)
            
            # 绘制到遮罩上
            cv2.fillPoly(refined_mask, [approx], 255)
        
        return refined_polygons, refined_mask

class CharacterDetector:
    def __init__(self, 
                 min_area=10, 
                 max_area=1000,
                 min_aspect_ratio=0.2, 
                 max_aspect_ratio=5,
                 min_solidity=0.3):
        """
        初始化字符检测器
        
        Args:
            min_area: 最小轮廓面积
            max_area: 最大轮廓面积
            min_aspect_ratio: 最小纵横比
            max_aspect_ratio: 最大纵横比
            min_solidity: 最小实心度（轮廓面积与凸包面积的比率）
        """
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_solidity = min_solidity
    
    def detect(self, image, text_mask=None):
        """
        检测图像中的单个字符
        
        Args:
            image: 输入图像 (BGR格式)
            text_mask: 可选的文本区域遮罩，用于限制检测范围
            
        Returns:
            字符轮廓列表，字符遮罩
        """
        # 转换为灰度图像
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 如果提供了文本遮罩，则仅在遮罩区域内检测
        if text_mask is not None:
            gray = cv2.bitwise_and(gray, gray, mask=text_mask)
        
        # 使用自适应阈值处理检测字符
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 应用形态学操作以改善字符形状
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)
        
        # 查找轮廓
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # 创建字符遮罩
        char_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # 过滤轮廓以找到字符
        char_contours = []
        for contour in contours:
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue
            
            # 获取边界框
            x, y, w, h = cv2.boundingRect(contour)
            
            # 计算纵横比
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # 计算实心度（轮廓面积与边界框面积的比率）
            rect_area = w * h
            solidity = float(area) / rect_area if rect_area > 0 else 0
            
            # 字符通常具有适中的纵横比和较高的实心度
            if (self.min_aspect_ratio < aspect_ratio < self.max_aspect_ratio and 
                solidity > self.min_solidity):
                # 检查是否为文本特征而非角色特征
                if self._is_character(contour, gray):
                    # 添加到字符轮廓列表
                    char_contours.append(contour)
                    
                    # 绘制到遮罩上
                    cv2.drawContours(char_mask, [contour], -1, 255, -1)
        
        return char_contours, char_mask
    
    def _is_character(self, contour, gray_image):
        """
        判断轮廓是否为字符而非角色特征
        
        Args:
            contour: 轮廓
            gray_image: 灰度图像
            
        Returns:
            布尔值，表示是否为字符
        """
        # 获取轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)
        
        # 计算轮廓的周长与面积的比率
        # 字符通常具有较大的周长/面积比
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        perimeter_area_ratio = perimeter / area if area > 0 else 0
        
        # 计算轮廓的圆形度
        # 圆形度 = 4π * 面积 / 周长²
        circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # 计算轮廓的凸度
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0
        
        # 字符通常具有适中的圆形度和凸度
        is_char_by_shape = (0.1 < circularity < 0.9) and (convexity > 0.7)
        
        # 检查轮廓是否与其他非字符特征连接
        is_isolated = self._is_isolated_feature(contour, gray_image)
        
        # 检查轮廓的梯度特征
        has_text_gradient = self._has_text_gradient(contour, gray_image)
        
        return is_char_by_shape and is_isolated and has_text_gradient
    
    def _is_isolated_feature(self, contour, gray_image):
        """
        判断轮廓是否为孤立特征（字符通常是孤立的）
        
        Args:
            contour: 轮廓
            gray_image: 灰度图像
            
        Returns:
            布尔值，表示是否为孤立特征
        """
        # 获取轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)
        
        # 扩展边界框以检查周围区域
        padding = 5
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(gray_image.shape[1], x + w + padding)
        y2 = min(gray_image.shape[0], y + h + padding)
        
        # 创建轮廓遮罩
        mask = np.zeros_like(gray_image)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # 创建扩展遮罩
        expanded_mask = np.zeros_like(gray_image)
        expanded_rect = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
        cv2.fillPoly(expanded_mask, [expanded_rect], 255)
        
        # 创建边界遮罩（扩展区域减去轮廓区域）
        boundary_mask = cv2.bitwise_and(expanded_mask, cv2.bitwise_not(mask))
        
        # 计算边界区域的平均灰度值
        boundary_mean = cv2.mean(gray_image, mask=boundary_mask)[0]
        
        # 计算轮廓区域的平均灰度值
        contour_mean = cv2.mean(gray_image, mask=mask)[0]
        
        # 计算对比度
        contrast = abs(contour_mean - boundary_mean)
        
        # 高对比度表示孤立特征
        return contrast > 30
    
    def _has_text_gradient(self, contour, gray_image):
        """
        判断轮廓是否具有文本梯度特征
        
        Args:
            contour: 轮廓
            gray_image: 灰度图像
            
        Returns:
            布尔值，表示是否具有文本梯度特征
        """
        # 获取轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)
        
        # 确保边界框在图像范围内
        if x < 0 or y < 0 or x + w >= gray_image.shape[1] or y + h >= gray_image.shape[0]:
            return False
        
        # 提取轮廓区域
        roi = gray_image[y:y+h, x:x+w]
        
        # 计算梯度
        gradient_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # 计算平均梯度幅值
        mean_gradient = np.mean(gradient_magnitude)
        
        # 计算梯度方向直方图
        gradient_direction = np.arctan2(gradient_y, gradient_x) * 180 / np.pi
        hist, _ = np.histogram(gradient_direction, bins=8, range=(-180, 180))
        hist = hist / np.sum(hist)
        
        # 计算方向熵
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # 文本通常具有较高的平均梯度和适中的方向熵
        return mean_gradient > 20 and 1.5 < entropy < 3.0

def test_polygon_detection():
    """
    测试多边形文本检测和字符检测功能
    """
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
    
    # 初始化多边形文本检测器
    poly_detector = PolygonTextDetector(
        min_area=50,
        min_aspect_ratio=0.1,
        max_aspect_ratio=10,
        epsilon_factor=0.02
    )
    
    # 初始化字符检测器
    char_detector = CharacterDetector(
        min_area=10,
        max_area=1000,
        min_aspect_ratio=0.2,
        max_aspect_ratio=5,
        min_solidity=0.3
    )
    
    # 检测文本多边形
    try:
        # 检测初始多边形
        polygons, poly_mask = poly_detector.detect(image)
        
        # 细化多边形
        refined_polygons, refined_poly_mask = poly_detector.refine_polygons(image, polygons)
        
        # 检测字符
        char_contours, char_mask = char_detector.detect(image, refined_poly_mask)
        
        # 绘制结果
        poly_result = image.copy()
        for poly in polygons:
            cv2.polylines(poly_result, [poly], True, (0, 0, 255), 2)
        
        refined_result = image.copy()
        for poly in refined_polygons:
            cv2.polylines(refined_result, [poly], True, (0, 255, 0), 2)
        
        char_result = image.copy()
        cv2.drawContours(char_result, char_contours, -1, (255, 0, 0), 2)
        
        # 保存结果
        filename = os.path.basename(image_path)
        base_name, ext = os.path.splitext(filename)
        
        # 保存多边形检测结果
        poly_result_path = os.path.join(output_dir, f"{base_name}_poly{ext}")
        cv2.imwrite(poly_result_path, poly_result)
        
        # 保存多边形遮罩
        poly_mask_path = os.path.join(output_dir, f"{base_name}_poly_mask{ext}")
        cv2.imwrite(poly_mask_path, poly_mask)
        
        # 保存细化多边形结果
        refined_result_path = os.path.join(output_dir, f"{base_name}_refined_poly{ext}")
        cv2.imwrite(refined_result_path, refined_result)
        
        # 保存细化多边形遮罩
        refined_mask_path = os.path.join(output_dir, f"{base_name}_refined_poly_mask{ext}")
        cv2.imwrite(refined_mask_path, refined_poly_mask)
        
        # 保存字符检测结果
        char_result_path = os.path.join(output_dir, f"{base_name}_char{ext}")
        cv2.imwrite(char_result_path, char_result)
        
        # 保存字符遮罩
        char_mask_path = os.path.join(output_dir, f"{base_name}_char_mask{ext}")
        cv2.imwrite(char_mask_path, char_mask)
        
        print(f"检测到 {len(polygons)} 个文本多边形")
        print(f"细化后有 {len(refined_polygons)} 个文本多边形")
        print(f"检测到 {len(char_contours)} 个字符")
        print(f"结果已保存到 {output_dir}")
    except Exception as e:
        print(f"检测过程中出现错误: {e}")

if __name__ == "__main__":
    test_polygon_detection()
