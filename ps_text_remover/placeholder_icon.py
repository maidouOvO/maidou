# 创建一个简单的图标文件
# 在Windows环境中运行此脚本生成icon.ico文件

from PIL import Image, ImageDraw

# 创建一个128x128的图像，背景为白色
img = Image.new('RGBA', (128, 128), color=(255, 255, 255, 0))
draw = ImageDraw.Draw(img)

# 绘制一个蓝色圆形
draw.ellipse((10, 10, 118, 118), fill=(65, 105, 225, 255))

# 绘制一个白色的"T"字母（代表Text）
draw.rectangle((48, 30, 80, 40), fill=(255, 255, 255, 255))
draw.rectangle((58, 40, 70, 90), fill=(255, 255, 255, 255))

# 保存为PNG
img.save('icon.png')

# 转换为ICO
img.save('icon.ico', format='ICO', sizes=[(128, 128)])

print("图标文件已创建：icon.ico")
