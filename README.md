# OCR和Photoshop自动化工具

这个工具使用Tesseract OCR识别图片中的文字，并自动使用Photoshop的套索工具框选特定类型的文字内容，然后应用修复工具。

## 功能特点

- 使用Tesseract OCR识别图片中的文字
- 根据以下规则自动分类文字：
  - 标题文字（自动框选）
  - 出现在物品、角色内的非连续文字（不框选）
  - 连续一句及以上、字符大小接近、颜色相似的文字内容（自动框选）
- 自动控制Photoshop的套索工具进行框选
- 使用Photoshop的修复工具进行修复

## 安装要求

- Python 3.6+
- Tesseract OCR
- Adobe Photoshop
- 以下Python库：
  - pytesseract
  - pillow
  - numpy
  - opencv-python
  - pyautogui

## 安装

1. 安装Tesseract OCR：
   - Windows: 从[这里](https://github.com/UB-Mannheim/tesseract/wiki)下载并安装
   - macOS: `brew install tesseract`
   - Linux: `sudo apt install tesseract-ocr`

2. 安装Python依赖：
   ```
   pip install pytesseract pillow numpy opencv-python pyautogui
   ```

## 使用方法

基本用法：

```
python ocr_photoshop_automation.py 图片路径
```

高级选项：

```
python ocr_photoshop_automation.py 图片路径 --width 800 --height 1280 --min-title-font-size 20 --min-continuous-text-length 10 --visualize 可视化输出路径.jpg
```

参数说明：
- `--width`：图像处理宽度（默认：800）
- `--height`：图像处理高度（默认：1280）
- `--min-title-font-size`：标题文字的最小字体大小（默认：20）
- `--min-continuous-text-length`：连续文本的最小长度（默认：10）
- `--min-text-confidence`：文字识别的最小置信度（默认：70）
- `--debug`：启用调试模式
- `--visualize`：保存可视化结果的路径
- `--no-ps`：跳过Photoshop自动化（仅执行OCR和分类）

## 注意事项

1. 使用前请确保Photoshop已打开
2. 该工具使用pyautogui控制Photoshop，请在运行过程中不要移动鼠标或使用键盘
3. 不同版本的Photoshop可能有不同的快捷键，可能需要调整代码中的快捷键设置

## 示例

```
python ocr_photoshop_automation.py 示例图片.jpg --visualize 结果.jpg
```

这将处理"示例图片.jpg"，识别并分类文字，将可视化结果保存到"结果.jpg"，并尝试在Photoshop中自动框选和修复文字区域。
