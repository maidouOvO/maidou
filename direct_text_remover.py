import os
import cv2
import numpy as np

class DirectTextRemover:
    """
    直接针对用户图像中的大字体文本进行检测和去除
    使用更激进的图像处理技术，专门针对大字体文本
    """
    def __init__(self, 
                 inpaint_method=cv2.INPAINT_TELEA,
                 dilation_kernel_size=7):
        """
        初始化直接文本去除器
        
        Args:
            inpaint_method: 图像修复方法
            dilation_kernel_size: 膨胀核大小
        """
        self.inpaint_method = inpaint_method
        self.dilation_kernel_size = dilation_kernel_size
    
    def _create_text_mask(self, image):
        """
        创建文本遮罩，专门针对图像中间的大字体文本
        
        Args:
            image: 输入图像
            
        Returns:
            文本遮罩
        """
        # 转换为灰度图像
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 获取图像尺寸
        h, w = gray.shape[:2]
        
        # 创建一个空白遮罩
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 计算图像中心区域（大约是图像的中间三分之一）
        center_x = w // 2
        center_y = h // 2
        center_w = w // 3
        center_h = h // 3
        
        # 定义中心区域的边界
        x1 = center_x - center_w // 2
        y1 = center_y - center_h // 2
        x2 = center_x + center_w // 2
        y2 = center_y + center_h // 2
        
        # 提取中心区域
        center_region = gray[y1:y2, x1:x2]
        
        # 对中心区域应用多种二值化方法
        # 1. 全局阈值
        _, binary_global = cv2.threshold(center_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 2. 自适应阈值 - 大块大小，适用于大字体文本
        binary_adaptive = cv2.adaptiveThreshold(
            center_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 35, 10
        )
        
        # 3. 固定阈值（针对大字体）
        _, binary_fixed = cv2.threshold(center_region, 180, 255, cv2.THRESH_BINARY_INV)
        
        # 合并二值化结果
        binary_combined = cv2.bitwise_or(binary_global, binary_adaptive)
        binary_combined = cv2.bitwise_or(binary_combined, binary_fixed)
        
        # 应用形态学操作以改善文本形状
        kernel = np.ones((5, 5), np.uint8)
        binary_combined = cv2.dilate(binary_combined, kernel, iterations=2)
        binary_combined = cv2.erode(binary_combined, kernel, iterations=1)
        
        # 应用闭操作以填充大字体文本中的空洞
        close_kernel = np.ones((15, 15), np.uint8)
        binary_combined = cv2.morphologyEx(binary_combined, cv2.MORPH_CLOSE, close_kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(
            binary_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # 过滤轮廓并填充到遮罩中
        for contour in contours:
            # 计算轮廓的面积
            area = cv2.contourArea(contour)
            
            # 过滤掉太小的轮廓
            if area < 100:
                continue
            
            # 获取轮廓的边界框
            x, y, w, h = cv2.boundingRect(contour)
            
            # 计算纵横比
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # 过滤掉纵横比不合理的轮廓
            if aspect_ratio < 0.1 or aspect_ratio > 15:
                continue
            
            # 调整轮廓坐标到原始图像坐标系
            contour[:, :, 0] += x1
            contour[:, :, 1] += y1
            
            # 填充轮廓到遮罩中
            cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # 应用膨胀操作以确保完全覆盖文本
        if self.dilation_kernel_size > 0:
            kernel = np.ones((self.dilation_kernel_size, self.dilation_kernel_size), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)
        
        return mask
    
    def _create_manual_mask(self, image):
        """
        手动创建遮罩，专门针对用户图像中间的大字体文本
        
        Args:
            image: 输入图像
            
        Returns:
            手动创建的遮罩
        """
        # 获取图像尺寸
        h, w = image.shape[:2]
        
        # 创建一个空白遮罩
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 根据图像尺寸，手动定义文本区域的多边形
        # 这些坐标是根据用户图像中大字体文本的位置估计的
        # 对于不同的图像，这些坐标需要调整
        text_polygon = np.array([
            [w//3, h//3],
            [2*w//3, h//3],
            [2*w//3, 2*h//3],
            [w//3, 2*h//3]
        ], dtype=np.int32)
        
        # 填充多边形区域
        cv2.fillPoly(mask, [text_polygon], 255)
        
        return mask
    
    def remove_text(self, image):
        """
        去除图像中的文本，专门针对用户图像中间的大字体文本
        
        Args:
            image: 输入图像
            
        Returns:
            去除文本后的图像，文本遮罩
        """
        # 转换为BGR格式（如果不是）
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # 创建文本遮罩
        text_mask = self._create_text_mask(image)
        
        # 创建手动遮罩
        manual_mask = self._create_manual_mask(image)
        
        # 合并两个遮罩
        combined_mask = cv2.bitwise_or(text_mask, manual_mask)
        
        # 使用图像修复去除文本
        result = cv2.inpaint(image, combined_mask, 5, self.inpaint_method)
        
        return result, combined_mask

def test_direct_text_removal():
    """
    测试直接文本去除功能，专门针对用户图像中间的大字体文本
    """
    # 设置参数
    image_path = 'data/test1.jpg'
    output_dir = 'results'
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 确保数据目录存在
    if not os.path.exists('data'):
        os.makedirs('data')
        # 将用户图像复制到数据目录
        user_image_path = '~/attachments/195ee0f4-9c15-42f3-848b-146d23f62517/test1.jpg'
        os.system(f'cp {user_image_path} data/test1.jpg')
    
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
    
    # 初始化直接文本去除器
    remover = DirectTextRemover(
        inpaint_method=cv2.INPAINT_TELEA,
        dilation_kernel_size=7
    )
    
    # 去除文本
    try:
        print("开始处理用户图像...")
        result, mask = remover.remove_text(image)
        
        # 保存结果
        filename = os.path.basename(image_path)
        base_name, ext = os.path.splitext(filename)
        
        # 保存原始图像
        original_path = os.path.join(output_dir, f"{base_name}_direct_original{ext}")
        cv2.imwrite(original_path, image)
        
        # 保存去除文本后的图像
        result_path = os.path.join(output_dir, f"{base_name}_direct_result{ext}")
        cv2.imwrite(result_path, result)
        
        # 保存文本遮罩
        mask_path = os.path.join(output_dir, f"{base_name}_direct_mask{ext}")
        cv2.imwrite(mask_path, mask)
        
        print(f"用户图像 {filename} 处理完成")
        print(f"结果已保存到 {output_dir}")
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {e}")

if __name__ == "__main__":
    test_direct_text_removal()
