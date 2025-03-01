import os
import cv2
import numpy as np
from text_removal.text_detector import TextDetector
from polygon_text_detector import PolygonTextDetector, CharacterDetector

class OCRTextRemover:
    """
    使用CRAFT检测文字区域，然后通过多边形和单字符检测方法去除文字形状
    """
    def __init__(self, 
                 text_threshold=0.7,
                 link_threshold=0.4,
                 low_text=0.4,
                 cuda=False,
                 padding=3,
                 inpaint_method=cv2.INPAINT_TELEA,
                 min_area=100,
                 min_aspect_ratio=0.2,
                 max_aspect_ratio=8,
                 epsilon_factor=0.02,
                 char_min_area=20,
                 char_max_area=2000,  # 增大最大字符面积以处理大字体
                 char_min_aspect_ratio=0.2,
                 char_max_aspect_ratio=5,
                 char_min_solidity=0.3,
                 dilation_kernel_size=3):
        """
        初始化OCR文本去除器
        
        Args:
            text_threshold: 文本置信度阈值
            link_threshold: 链接置信度阈值
            low_text: 文本低边界分数
            cuda: 是否使用CUDA
            padding: 文本区域填充大小
            inpaint_method: 图像修复方法
            min_area: 最小多边形面积
            min_aspect_ratio: 最小纵横比
            max_aspect_ratio: 最大纵横比
            epsilon_factor: 多边形近似的epsilon因子
            char_min_area: 最小字符面积
            char_max_area: 最大字符面积
            char_min_aspect_ratio: 最小字符纵横比
            char_max_aspect_ratio: 最大字符纵横比
            char_min_solidity: 最小字符实心度
            dilation_kernel_size: 膨胀核大小
        """
        # 初始化CRAFT文本检测器
        self.text_detector = TextDetector(
            text_threshold=text_threshold,
            low_text=low_text,
            link_threshold=link_threshold,
            cuda=cuda
        )
        
        # 初始化多边形文本检测器
        self.poly_detector = PolygonTextDetector(
            min_area=min_area,
            min_aspect_ratio=min_aspect_ratio,
            max_aspect_ratio=max_aspect_ratio,
            epsilon_factor=epsilon_factor
        )
        
        # 初始化字符检测器
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
    
    def _detect_text_regions(self, image):
        """
        使用改进的文本检测方法检测文本区域，特别优化用于检测大字体文本
        
        Args:
            image: 输入图像
            
        Returns:
            文本区域遮罩，文本框列表
        """
        # 使用改进的文本检测器检测文本区域
        boxes, polys, _ = self.text_detector.detect(image)
        
        # 创建文本区域遮罩
        region_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # 填充多边形区域
        for poly in polys:
            if poly is not None:
                cv2.fillPoly(region_mask, [poly.astype(np.int32)], 255)
        
        # 填充边界框区域
        for box in boxes:
            if box is not None:
                cv2.fillPoly(region_mask, [box.astype(np.int32)], 255)
        
        # 应用膨胀操作以扩展遮罩
        if self.padding > 0:
            kernel = np.ones((self.padding, self.padding), np.uint8)
            region_mask = cv2.dilate(region_mask, kernel, iterations=3)  # 增加迭代次数
        
        # 添加额外的处理步骤，用于检测大字体文本
        # 1. 使用形态学闭操作填充文本区域内的空洞
        close_kernel = np.ones((15, 15), np.uint8)
        region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, close_kernel)
        
        # 2. 连接相近的文本区域
        connect_kernel = np.ones((20, 5), np.uint8)  # 水平连接
        region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, connect_kernel)
        
        connect_kernel = np.ones((5, 20), np.uint8)  # 垂直连接
        region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, connect_kernel)
        
        # 3. 再次应用膨胀操作，确保覆盖整个文本区域
        region_mask = cv2.dilate(region_mask, kernel, iterations=1)
            
        return region_mask, boxes
    
    def _detect_text_shapes(self, image, region_mask):
        """
        在文本区域内检测文本形状
        
        Args:
            image: 输入图像
            region_mask: 文本区域遮罩
            
        Returns:
            文本形状遮罩
        """
        # 转换为灰度图像
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 在文本区域内应用多种二值化方法
        # 1. 全局阈值
        _, binary_global = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary_global = cv2.bitwise_and(binary_global, region_mask)
        
        # 2. 自适应阈值
        binary_adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 15, 5
        )
        binary_adaptive = cv2.bitwise_and(binary_adaptive, region_mask)
        
        # 3. 固定阈值（针对大字体）
        _, binary_fixed = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        binary_fixed = cv2.bitwise_and(binary_fixed, region_mask)
        
        # 合并二值化结果
        binary_combined = cv2.bitwise_or(binary_global, binary_adaptive)
        binary_combined = cv2.bitwise_or(binary_combined, binary_fixed)
        
        # 应用形态学操作以改善文本形状，特别是大字体文本
        kernel = np.ones((5, 5), np.uint8)  # 增大核大小
        binary_combined = cv2.dilate(binary_combined, kernel, iterations=2)  # 增加迭代次数
        binary_combined = cv2.erode(binary_combined, kernel, iterations=1)
        
        # 应用闭操作以填充大字体文本中的空洞
        close_kernel = np.ones((15, 15), np.uint8)
        binary_combined = cv2.morphologyEx(binary_combined, cv2.MORPH_CLOSE, close_kernel)
        
        # 使用多边形检测器检测文本形状
        polygons, poly_mask = self.poly_detector.detect(image)
        poly_mask = cv2.bitwise_and(poly_mask, region_mask)
        
        # 使用字符检测器检测单个字符
        char_contours, char_mask = self.char_detector.detect(image, region_mask)
        
        # 合并多边形遮罩和字符遮罩
        shape_mask = cv2.bitwise_or(poly_mask, char_mask)
        
        # 合并所有遮罩
        combined_mask = cv2.bitwise_or(shape_mask, binary_combined)
        
        # 应用膨胀操作以确保完全覆盖文本
        kernel = np.ones((self.dilation_kernel_size, self.dilation_kernel_size), np.uint8)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
        
        return combined_mask
    
    def _filter_character_features(self, mask, image):
        """
        过滤掉角色特征，只保留文本特征
        
        Args:
            mask: 输入遮罩
            image: 输入图像
            
        Returns:
            过滤后的遮罩
        """
        # 转换为灰度图像
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 查找轮廓
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # 创建新的遮罩，仅包含文本特征
        filtered_mask = np.zeros_like(mask)
        
        for contour in contours:
            # 计算轮廓的边界框
            x, y, w, h = cv2.boundingRect(contour)
            
            # 计算轮廓的面积
            area = cv2.contourArea(contour)
            
            # 计算轮廓的周长
            perimeter = cv2.arcLength(contour, True)
            
            # 计算轮廓的复杂度（周长/面积）
            complexity = perimeter / area if area > 0 else 0
            
            # 计算轮廓的纵横比
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # 计算轮廓的凸包
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            
            # 计算轮廓的实心度（轮廓面积/凸包面积）
            solidity = area / hull_area if hull_area > 0 else 0
            
            # 检查是否为文本特征
            is_text_feature = (
                # 文本通常具有适中的纵横比
                0.2 < aspect_ratio < 8 and
                # 文本通常具有较高的复杂度
                complexity > 0.1 and
                # 文本通常具有适中的实心度
                0.3 < solidity < 0.9 and
                # 过滤掉过大或过小的轮廓
                20 < area < 5000
            )
            
            # 检查是否与其他特征连接
            is_isolated = self._is_isolated_feature(contour, gray)
            
            if is_text_feature and is_isolated:
                # 如果是文本特征，则添加到过滤后的遮罩
                cv2.drawContours(filtered_mask, [contour], -1, 255, -1)
        
        return filtered_mask
    
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
        
    def remove_text(self, image):
        """
        去除图像中的文本
        
        Args:
            image: 输入图像
            
        Returns:
            去除文本后的图像，文本区域遮罩，文本形状遮罩
        """
        # 转换为BGR格式（如果不是）
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # 检测文本区域
        region_mask, _ = self._detect_text_regions(image)
        
        # 检测文本形状
        shape_mask = self._detect_text_shapes(image, region_mask)
        
        # 过滤掉角色特征
        filtered_mask = self._filter_character_features(shape_mask, image)
        
        # 使用图像修复去除文本
        result = cv2.inpaint(image, filtered_mask, 3, self.inpaint_method)
        
        return result, region_mask, filtered_mask

def test_ocr_text_removal():
    """
    测试OCR文本去除功能
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
    
    # 初始化OCR文本去除器
    remover = OCRTextRemover(
        text_threshold=0.6,  # 降低文本阈值以检测更多文本
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
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {e}")

if __name__ == "__main__":
    test_ocr_text_removal()
