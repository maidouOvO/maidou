import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# 创建一个白色背景图像
img = np.ones((512, 512, 3), dtype=np.uint8) * 255

# 转换为PIL图像以便添加文字
pil_img = Image.fromarray(img)
draw = ImageDraw.Draw(pil_img)

# 添加文字
text = '这是一个测试文本'
position = (100, 200)
text_color = (0, 0, 0)  # 黑色文字

# 尝试使用默认字体
try:
    draw.text(position, text, fill=text_color)
except Exception as e:
    print(f'使用默认字体失败: {e}')
    # 如果默认字体失败，使用简单的矩形代替文字
    for i in range(len(text)):
        x = position[0] + i * 30
        y = position[1]
        draw.rectangle([(x, y), (x + 20, y + 30)], fill=text_color)

# 转换回OpenCV格式并保存
img = np.array(pil_img)
cv2.imwrite('data/sample.jpg', img)
print('已创建测试图像: data/sample.jpg')
