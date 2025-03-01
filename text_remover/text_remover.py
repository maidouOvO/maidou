import cv2
import numpy as np
from skimage import morphology

class TextRemover:
    def __init__(self):
        pass
    
    def create_expanded_mask(self, image, polys, expansion=5):
        """创建扩展的初始遮罩"""
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # 绘制多边形区域
        for poly in polys:
            if poly is not None and len(poly) > 0:
                # 确保多边形点是整数
                pts = np.array(poly, np.int32)
                if pts.size > 0:  # 确保有点
                    pts = pts.reshape((-1, 1, 2))
                    cv2.fillPoly(mask, [pts], 255)
                    
        # 打印调试信息
        print(f"创建的遮罩中白色像素数量: {np.sum(mask == 255)}")
        
        # 扩展遮罩
        kernel = np.ones((expansion, expansion), np.uint8)
        expanded_mask = cv2.dilate(mask, kernel, iterations=1)
        
        return expanded_mask
    
    def refine_text_mask(self, image, expanded_mask, block_size=11, c=2):
        """在扩展遮罩内精确识别文字形状"""
        # 提取遮罩区域
        masked_image = cv2.bitwise_and(image, image, mask=expanded_mask)
        
        # 转换为灰度图
        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        
        # 自适应阈值处理，处理复杂背景
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, block_size, c)
        
        # 只保留扩展遮罩内的区域
        refined_mask = cv2.bitwise_and(binary, binary, mask=expanded_mask)
        
        # 处理中空字体问题
        kernel = np.ones((3, 3), np.uint8)
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
        
        # 移除小噪点
        refined_mask = morphology.remove_small_objects(refined_mask.astype(bool), 
                                                     min_size=50).astype(np.uint8) * 255
        
        return refined_mask
    
    def optimize_hollow_text_detection(self, image, expanded_mask):
        """优化中空大字体的检测"""
        # 提取遮罩区域
        masked_image = cv2.bitwise_and(image, image, mask=expanded_mask)
        
        # 转换为灰度图
        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        
        # 使用多种阈值方法
        # 1. 自适应阈值 - 适用于复杂背景
        binary1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # 2. Otsu阈值 - 适用于双峰图像
        _, binary2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 3. 固定阈值 - 适用于高对比度
        _, binary3 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # 合并结果
        binary = cv2.bitwise_or(binary1, binary2)
        binary = cv2.bitwise_or(binary, binary3)
        
        # 只保留扩展遮罩内的区域
        refined_mask = cv2.bitwise_and(binary, binary, mask=expanded_mask)
        
        # 处理中空字体问题
        kernel_close = np.ones((5, 5), np.uint8)  # 更大的核用于闭操作
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel_close)
        
        # 填充孔洞
        contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # 只填充较大的轮廓
                cv2.drawContours(refined_mask, [contour], 0, 255, -1)
        
        # 移除小噪点
        refined_mask = morphology.remove_small_objects(refined_mask.astype(bool), 
                                                     min_size=50).astype(np.uint8) * 255
        
        return refined_mask
    
    def remove_text(self, image, refined_mask, method='inpaint'):
        """使用遮罩移除文字"""
        if method == 'inpaint':
            # 使用OpenCV的inpaint方法填充文字区域
            result = cv2.inpaint(image, refined_mask, 3, cv2.INPAINT_TELEA)
        else:
            # 简单的背景填充（备选方法）
            result = image.copy()
            result[refined_mask == 255] = [255, 255, 255]  # 白色填充
        
        return result
