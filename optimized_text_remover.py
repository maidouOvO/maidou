import os
import cv2
import numpy as np
import math
from simple_text_detector import SimpleTextDetector
from polygon_text_detector import PolygonTextDetector, CharacterDetector

class OptimizedTextRemover:
    def __init__(self, 
                 padding=3,  # 减小填充大小以避免过度扩展
                 inpaint_method=cv2.INPAINT_TELEA,
                 min_area=80,  # 增加最小面积以过滤小噪点
                 min_aspect_ratio=0.2,  # 调整纵横比范围以更好地匹配文本
                 max_aspect_ratio=8,
                 epsilon_factor=0.02,
                 char_min_area=15,  # 增加最小字符面积
                 char_max_area=800,  # 减小最大字符面积以避免误检测大型角色特征
                 char_min_aspect_ratio=0.3,  # 调整字符纵横比
                 char_max_aspect_ratio=4,
                 char_min_solidity=0.4,  # 增加最小实心度以更好地匹配文本字符
                 dilation_kernel_size=2,  # 减小膨胀核大小以避免过度扩展
                 gradient_threshold=20,  # 梯度阈值，用于区分文本和非文本
                 connectivity_threshold=0.15,  # 连通性阈值，用于区分孤立文本和连接特征
                 character_detection_threshold=127):  # 字符检测阈值
        """
        初始化优化的文字去除器
        
        Args:
            padding: 文本区域填充大小，用于创建扩展遮罩
            inpaint_method: 图像修复方法，可选cv2.INPAINT_TELEA或cv2.INPAINT_NS
            min_area: 最小多边形面积
            min_aspect_ratio: 最小纵横比
            max_aspect_ratio: 最大纵横比
            epsilon_factor: 多边形近似的epsilon因子
            char_min_area: 最小字符面积
            char_max_area: 最大字符面积
            char_min_aspect_ratio: 最小字符纵横比
            char_max_aspect_ratio: 最大字符纵横比
            char_min_solidity: 最小字符实心度
            dilation_kernel_size: 膨胀核大小，用于扩展遮罩
            gradient_threshold: 梯度阈值，用于区分文本和非文本
            connectivity_threshold: 连通性阈值，用于区分孤立文本和连接特征
            character_detection_threshold: 字符检测阈值
        """
        self.simple_detector = SimpleTextDetector()
        self.poly_detector = PolygonTextDetector(
            min_area=min_area,
            min_aspect_ratio=min_aspect_ratio,
            max_aspect_ratio=max_aspect_ratio,
            epsilon_factor=epsilon_factor
        )
        self.char_detector = CharacterDetector(
            min_area=char_min_area,
            max_area=char_max_area,
            min_aspect_ratio=char_min_aspect_ratio,
            max_aspect_ratio=char_max_aspect_ratio,
            min_solidity=char_min_solidity
        )
        self.padding = padding
        self.inpaint_method = inpaint_method
        self.dilation_kernel_size = dilation_kernel_size
        self.gradient_threshold = gradient_threshold
        self.connectivity_threshold = connectivity_threshold
        self.character_detection_threshold = character_detection_threshold
    
    def _create_expanded_mask(self, image, polygons, boxes):
        """
        创建扩展遮罩，覆盖文本区域并添加一些填充
        
        Args:
            image: 输入图像
            polygons: 文本多边形列表
            boxes: 文本边界框列表
            
        Returns:
            扩展遮罩
        """
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # 添加多边形区域
        for poly in polygons:
            cv2.fillPoly(mask, [poly], 255)
        
        # 添加边界框区域
        for box in boxes:
            # 将边界框转换为整数坐标
            box = box.astype(np.int32)
            # 绘制填充多边形
            cv2.fillPoly(mask, [box], 255)
        
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
        
        # 检测字符
        char_contours, char_mask = self.char_detector.detect(image, expanded_mask)
        
        # 合并字符遮罩和精细遮罩
        combined_mask = cv2.bitwise_or(refined_mask, char_mask)
        
        # 应用小的膨胀以确保完全覆盖文本
        kernel = np.ones((self.dilation_kernel_size, self.dilation_kernel_size), np.uint8)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
        
        return combined_mask
    
    def _calculate_gradient_features(self, gray_image, contour):
        """
        计算轮廓区域的梯度特征
        
        Args:
            gray_image: 灰度图像
            contour: 轮廓
            
        Returns:
            梯度幅值均值，梯度方向熵
        """
        # 创建轮廓遮罩
        mask = np.zeros_like(gray_image)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # 计算Sobel梯度
        grad_x = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)
        
        # 计算梯度幅值和方向
        magnitude = cv2.magnitude(grad_x, grad_y)
        angle = cv2.phase(grad_x, grad_y, angleInDegrees=True)
        
        # 仅考虑轮廓区域内的梯度
        magnitude_roi = magnitude[mask > 0]
        angle_roi = angle[mask > 0]
        
        # 计算梯度幅值均值
        mean_magnitude = np.mean(magnitude_roi) if magnitude_roi.size > 0 else 0
        
        # 计算梯度方向熵
        if angle_roi.size > 0:
            # 将角度分为18个bin（每20度一个bin）
            hist, _ = np.histogram(angle_roi, bins=18, range=(0, 360))
            hist = hist / hist.sum() if hist.sum() > 0 else hist
            # 计算熵
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
        else:
            entropy = 0
        
        return mean_magnitude, entropy
    
    def _is_character_feature(self, contour, gray_image):
        """
        判断轮廓是否为角色特征而非文本
        
        Args:
            contour: 轮廓
            gray_image: 灰度图像
            
        Returns:
            布尔值，表示是否为角色特征
        """
        # 计算轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)
        
        # 计算轮廓的面积
        area = cv2.contourArea(contour)
        
        # 计算轮廓的周长
        perimeter = cv2.arcLength(contour, True)
        
        # 计算轮廓的复杂度（周长/面积）
        complexity = perimeter / area if area > 0 else 0
        
        # 计算轮廓的圆形度
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # 计算轮廓的凸包
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        
        # 计算轮廓的实心度（轮廓面积/凸包面积）
        solidity = area / hull_area if hull_area > 0 else 0
        
        # 计算梯度特征
        mean_magnitude, direction_entropy = self._calculate_gradient_features(gray_image, contour)
        
        # 检查轮廓是否与其他特征连接
        is_connected = self._is_connected_feature(contour, gray_image)
        
        # 角色特征通常具有以下特点：
        # 1. 较大的面积
        is_character_by_size = area > 500
        
        # 2. 较低的复杂度
        is_character_by_complexity = complexity < 0.2
        
        # 3. 较高的圆形度（接近圆形或椭圆形）
        is_character_by_circularity = circularity > 0.6
        
        # 4. 较高的实心度
        is_character_by_solidity = solidity > 0.8
        
        # 5. 较低的梯度幅值（文本通常具有较高的梯度）
        is_character_by_gradient = mean_magnitude < self.gradient_threshold
        
        # 6. 适中的方向熵（文本通常具有较低或较高的方向熵）
        is_character_by_entropy = 1.0 < direction_entropy < 3.0
        
        # 7. 与其他特征连接
        is_character_by_connectivity = is_connected
        
        # 综合判断
        # 如果满足多个条件，则更可能是角色特征
        character_score = sum([
            is_character_by_size,
            is_character_by_complexity,
            is_character_by_circularity,
            is_character_by_solidity,
            is_character_by_gradient,
            is_character_by_entropy,
            is_character_by_connectivity
        ])
        
        return character_score >= 3  # 满足至少3个条件
    
    def _is_connected_feature(self, contour, gray_image):
        """
        判断轮廓是否与其他特征连接
        
        Args:
            contour: 轮廓
            gray_image: 灰度图像
            
        Returns:
            布尔值，表示是否与其他特征连接
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
        
        # 创建扩展遮罩
        expanded_mask = np.zeros_like(gray_image)
        expanded_rect = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
        cv2.fillPoly(expanded_mask, [expanded_rect], 255)
        
        # 创建边界遮罩（扩展区域减去轮廓区域）
        boundary_mask = cv2.bitwise_and(expanded_mask, cv2.bitwise_not(mask))
        
        # 应用Canny边缘检测
        edges = cv2.Canny(gray_image, 50, 150)
        
        # 计算边界区域内的边缘像素数量
        boundary_edges = cv2.bitwise_and(edges, boundary_mask)
        edge_count = np.sum(boundary_edges > 0)
        
        # 计算边界区域的面积
        boundary_area = np.sum(boundary_mask > 0)
        
        # 计算边缘密度
        edge_density = edge_count / boundary_area if boundary_area > 0 else 0
        
        # 高边缘密度表示与其他特征连接
        return edge_density > self.connectivity_threshold
    
    def _is_part_of_object(self, contour, gray_image):
        """
        判断轮廓是否为对象的一部分（如车牌号码、标签等）
        
        Args:
            contour: 轮廓
            gray_image: 灰度图像
            
        Returns:
            布尔值，表示是否为对象的一部分
        """
        # 获取轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)
        
        # 扩展边界框以检查周围区域
        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(gray_image.shape[1], x + w + padding)
        y2 = min(gray_image.shape[0], y + h + padding)
        
        # 提取扩展区域
        roi = gray_image[y1:y2, x1:x2]
        
        # 应用阈值处理
        _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 查找轮廓
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # 如果周围区域有大型轮廓，则可能是对象的一部分
        for c in contours:
            area = cv2.contourArea(c)
            if area > 500:  # 大型轮廓
                return True
        
        return False
    
    def _filter_character_features(self, mask, gray_image):
        """
        过滤掉角色特征，只保留文本特征
        
        Args:
            mask: 输入遮罩
            gray_image: 灰度图像
            
        Returns:
            过滤后的遮罩
        """
        # 查找轮廓
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # 创建新的遮罩，仅包含文本特征
        filtered_mask = np.zeros_like(mask)
        
        for contour in contours:
            # 检查是否为角色特征
            if not self._is_character_feature(contour, gray_image):
                # 检查是否为对象的一部分
                if not self._is_part_of_object(contour, gray_image):
                    # 如果既不是角色特征也不是对象的一部分，则添加到过滤后的遮罩
                    cv2.drawContours(filtered_mask, [contour], -1, 255, -1)
        
        return filtered_mask
    
    def remove_text(self, image):
        """
        从图像中去除文本
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            去除文本后的图像，扩展遮罩，过滤后的遮罩
        """
        # 转换为灰度图像
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # 检测文本区域
        boxes, _ = self.simple_detector.detect(image)
        
        # 检测文本多边形
        polygons, _ = self.poly_detector.detect(image)
        
        # 创建扩展遮罩
        expanded_mask = self._create_expanded_mask(image, polygons, boxes)
        
        # 创建精细遮罩
        refined_mask = self._create_refined_mask(image, expanded_mask)
        
        # 过滤掉角色特征
        filtered_mask = self._filter_character_features(refined_mask, gray)
        
        # 使用精细遮罩进行图像修复
        result = cv2.inpaint(image, filtered_mask, 3, self.inpaint_method)
        
        return result, expanded_mask, filtered_mask

def test_optimized_text_removal():
    """
    测试优化的文本去除功能
    """
    # 设置参数
    data_dir = 'data'
    output_dir = 'results'
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 初始化优化的文本去除器
    remover = OptimizedTextRemover(
        padding=3,
        inpaint_method=cv2.INPAINT_TELEA,
        min_area=80,
        min_aspect_ratio=0.2,
        max_aspect_ratio=8,
        epsilon_factor=0.02,
        char_min_area=15,
        char_max_area=800,
        char_min_aspect_ratio=0.3,
        char_max_aspect_ratio=4,
        char_min_solidity=0.4,
        dilation_kernel_size=2,
        gradient_threshold=20,
        connectivity_threshold=0.15
    )
    
    # 获取所有测试图像
    test_images = [
        os.path.join(data_dir, 'sample.jpg'),
        os.path.join(data_dir, 'shapes_and_text.jpg'),
        os.path.join(data_dir, 'character_with_text.jpg'),
        os.path.join(data_dir, 'text_as_part_of_object.jpg'),
        os.path.join(data_dir, 'large_font_text.jpg')
    ]
    
    # 测试每个图像
    for image_path in test_images:
        # 加载图像
        if not os.path.isfile(image_path):
            print(f"找不到图像: {image_path}")
            continue
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            continue
        
        # 去除文本
        try:
            result, expanded_mask, filtered_mask = remover.remove_text(image)
            
            # 保存结果
            filename = os.path.basename(image_path)
            base_name, ext = os.path.splitext(filename)
            
            # 保存原始图像
            original_path = os.path.join(output_dir, f"{base_name}_optimized_original{ext}")
            cv2.imwrite(original_path, image)
            
            # 保存去除文本后的图像
            result_path = os.path.join(output_dir, f"{base_name}_optimized_result{ext}")
            cv2.imwrite(result_path, result)
            
            # 保存扩展遮罩
            expanded_mask_path = os.path.join(output_dir, f"{base_name}_optimized_expanded_mask{ext}")
            cv2.imwrite(expanded_mask_path, expanded_mask)
            
            # 保存过滤后的遮罩
            filtered_mask_path = os.path.join(output_dir, f"{base_name}_optimized_filtered_mask{ext}")
            cv2.imwrite(filtered_mask_path, filtered_mask)
            
            print(f"图像 {filename} 处理完成")
        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {e}")
    
    print("优化的文本去除测试完成")

if __name__ == "__main__":
    test_optimized_text_removal()
