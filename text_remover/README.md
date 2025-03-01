# 文本检测与移除工具

这个工具使用CRAFT进行文字区域识别，并通过OpenCV对区域中的文字进行去除。

## 特点

- 使用CRAFT进行精确的文字区域检测
- 两步遮罩方法：扩展初始遮罩和精细文字形状检测
- 优化的中空大字体检测算法
- 保留图像中的非文字元素（如角色特征）

## 安装

```bash
# 安装依赖
pip install numpy opencv-python-headless torch torchvision matplotlib scikit-image scipy pillow
```

## 使用方法

```bash
python main.py --image 输入图像路径 --output 输出图像路径
```

### 参数

- `--image`: 输入图像路径
- `--output`: 输出图像路径
- `--text_threshold`: 文本检测阈值 (默认: 0.7)
- `--link_threshold`: 文本连接阈值 (默认: 0.4)
- `--low_text`: 低文本阈值 (默认: 0.4)
- `--cuda`: 使用CUDA加速
- `--expansion`: 遮罩扩展大小 (默认: 5)
- `--block_size`: 自适应阈值块大小 (默认: 11)
- `--c`: 自适应阈值常数 (默认: 2)
- `--optimize_hollow`: 优化中空字体检测 (默认: True)

## 示例

```bash
python test.py
```

## 工作原理

1. 使用CRAFT模型检测图像中的文字区域
2. 创建扩展的初始遮罩覆盖文字区域
3. 在扩展遮罩内进行精细的文字形状检测
4. 使用OpenCV的inpaint方法填充文字区域

## 后续计划

- 集成lama-cleaner进行更高质量的文字去除
- 改进对复杂背景中文字的识别精度
- 优化大字体中空问题的处理
