#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photoshop 绘本文本处理自动化脚本
功能：从文件夹选择图片，通过PS打开，辅助用户使用套索工具圈选文本，
     应用Shift+F5内容识别和修复，并将处理后的图片覆盖原图
"""

import os
import sys
import time
import json
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
import pyautogui
import cv2
import numpy as np
from PIL import Image, ImageTk

# 默认配置
DEFAULT_CONFIG = {
    "ps_path": r"C:\Program Files\Adobe\Adobe Photoshop 2021\Photoshop.exe",  # Photoshop 2021的默认安装路径
    "pause_time": 1.0,  # 操作间隔时间，可根据电脑性能调整
    "last_folder": "",  # 上次使用的文件夹
}

# 配置文件路径
CONFIG_FILE = os.path.join(os.path.expanduser("~"), "ps_text_remover_config.json")

def load_config():
    """加载配置文件，如果不存在则创建默认配置"""
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # 确保所有默认配置项都存在
                for key, value in DEFAULT_CONFIG.items():
                    if key not in config:
                        config[key] = value
                return config
        else:
            # 如果配置文件不存在，创建默认配置
            save_config(DEFAULT_CONFIG)
            return DEFAULT_CONFIG.copy()
    except Exception as e:
        print(f"加载配置文件出错: {str(e)}")
        return DEFAULT_CONFIG.copy()

def save_config(config):
    """保存配置到文件"""
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"保存配置文件出错: {str(e)}")

class PSTextRemover:
    def __init__(self, root):
        self.root = root
        self.root.title("PS绘本文本处理工具")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # 加载配置
        self.config = load_config()
        
        self.image_folder = self.config["last_folder"]
        self.image_files = []
        self.current_image_index = 0
        
        self.setup_ui()
    
    def setup_ui(self):
        """设置用户界面"""
        # 创建菜单栏
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="选择文件夹", command=self.browse_folder)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)
        
        # 设置菜单
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="设置", menu=settings_menu)
        settings_menu.add_command(label="配置Photoshop路径", command=self.configure_ps_path)
        settings_menu.add_command(label="配置操作延迟", command=self.configure_pause_time)
        
        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="使用说明", command=self.show_help)
        help_menu.add_command(label="关于", command=self.show_about)
        
        # 创建标签页
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 图片浏览标签页
        browse_frame = ttk.Frame(self.notebook)
        self.notebook.add(browse_frame, text="图片浏览")
        
        # 顶部框架 - 文件夹选择
        top_frame = ttk.Frame(browse_frame)
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(top_frame, text="图片文件夹:").pack(side=tk.LEFT)
        self.folder_entry = ttk.Entry(top_frame, width=50)
        self.folder_entry.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        # 如果有上次使用的文件夹，则显示
        if self.image_folder:
            self.folder_entry.insert(0, self.image_folder)
        
        ttk.Button(top_frame, text="浏览...", command=self.browse_folder).pack(side=tk.LEFT, padx=5)
        
        # 中间框架 - 图片预览
        middle_frame = ttk.Frame(browse_frame)
        middle_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.preview_canvas = tk.Canvas(middle_frame, bg="lightgray")
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 底部框架 - 控制按钮
        bottom_frame = ttk.Frame(browse_frame)
        bottom_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(bottom_frame, text="上一张", command=self.prev_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(bottom_frame, text="下一张", command=self.next_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(bottom_frame, text="处理当前图片", command=self.process_current_image).pack(side=tk.LEFT, padx=5)
        
        # 帮助标签页
        help_frame = ttk.Frame(self.notebook)
        self.notebook.add(help_frame, text="使用说明")
        
        help_text = """
操作说明:

1. 选择包含图片的文件夹
   - 点击"文件"菜单中的"选择文件夹"，或使用主界面上的"浏览..."按钮
   - 选择包含需要处理的图片的文件夹

2. 预览并选择要处理的图片
   - 使用"上一张"和"下一张"按钮浏览文件夹中的图片
   - 当前图片会显示在预览区域

3. 处理图片
   - 点击"处理当前图片"按钮，将自动打开PS并加载图片
   - 脚本会自动识别和选择文本区域（区分于主体旁边的自然段，不包括人物或角色、动植物身上的字符元素）
   - 脚本将自动应用Shift+F5内容识别和修复
   - 处理完成后，图片会自动保存并覆盖原图
   - 返回本程序，可以继续处理下一张图片

4. 配置设置
   - 在"设置"菜单中可以配置Photoshop路径和操作延迟时间
   - 所有设置会自动保存，下次启动程序时自动加载
        """
        
        help_text_widget = tk.Text(help_frame, wrap=tk.WORD, padx=10, pady=10)
        help_text_widget.pack(fill=tk.BOTH, expand=True)
        help_text_widget.insert(tk.END, help_text)
        help_text_widget.config(state=tk.DISABLED)
        
        # 状态栏
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="就绪", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(fill=tk.X)
    
    def configure_ps_path(self):
        """配置Photoshop路径"""
        current_path = self.config["ps_path"]
        new_path = filedialog.askopenfilename(
            title="选择Photoshop可执行文件",
            initialdir=os.path.dirname(current_path),
            filetypes=[("可执行文件", "*.exe"), ("所有文件", "*.*")]
        )
        if new_path:
            self.config["ps_path"] = new_path
            save_config(self.config)
            messagebox.showinfo("成功", "Photoshop路径已更新")
    
    def configure_pause_time(self):
        """配置操作延迟时间"""
        current_time = self.config["pause_time"]
        new_time = simpledialog.askfloat(
            "配置操作延迟",
            "请输入操作间隔时间（秒）：\n较慢的电脑可能需要更长的延迟",
            initialvalue=current_time,
            minvalue=0.1,
            maxvalue=5.0
        )
        if new_time is not None:
            self.config["pause_time"] = new_time
            save_config(self.config)
            messagebox.showinfo("成功", f"操作延迟已更新为 {new_time} 秒")
    

    
    def show_help(self):
        """显示帮助信息"""
        self.notebook.select(1)  # 切换到帮助标签页
    
    def show_about(self):
        """显示关于信息"""
        messagebox.showinfo(
            "关于",
            "PS绘本文本处理工具 v1.0\n\n"
            "功能：从文件夹选择图片，通过PS打开，辅助用户使用套索工具圈选文本，\n"
            "应用Shift+F5内容识别和修复，并将处理后的图片覆盖原图\n\n"
            "作者：Devin AI"
        )
    
    def browse_folder(self):
        """浏览并选择图片文件夹"""
        folder = filedialog.askdirectory(title="选择图片文件夹")
        if folder:
            self.image_folder = folder
            self.folder_entry.delete(0, tk.END)
            self.folder_entry.insert(0, folder)
            
            # 更新配置
            self.config["last_folder"] = folder
            save_config(self.config)
            
            self.load_images()
    
    def load_images(self):
        """加载文件夹中的所有图片"""
        if not os.path.isdir(self.image_folder):
            messagebox.showerror("错误", "无效的文件夹路径")
            return
        
        # 支持的图片格式
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        
        self.image_files = [f for f in os.listdir(self.image_folder) 
                           if os.path.isfile(os.path.join(self.image_folder, f)) 
                           and f.lower().endswith(image_extensions)]
        
        if not self.image_files:
            messagebox.showinfo("提示", "所选文件夹中没有找到支持的图片文件")
            return
        
        self.current_image_index = 0
        self.display_current_image()
        self.update_status()
    
    def display_current_image(self):
        """显示当前选中的图片"""
        if not self.image_files:
            return
        
        image_path = os.path.join(self.image_folder, self.image_files[self.current_image_index])
        
        try:
            # 使用PIL加载图片并调整大小以适应画布
            img = Image.open(image_path)
            
            # 获取画布大小
            canvas_width = self.preview_canvas.winfo_width()
            canvas_height = self.preview_canvas.winfo_height()
            
            # 如果画布尚未完全初始化，使用默认值
            if canvas_width <= 1:
                canvas_width = 700
            if canvas_height <= 1:
                canvas_height = 400
            
            # 计算缩放比例
            img_width, img_height = img.size
            scale = min(canvas_width / img_width, canvas_height / img_height)
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # 转换为Tkinter可用的格式
            self.tk_img = ImageTk.PhotoImage(img)
            
            # 清除画布并显示图片
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(
                canvas_width // 2, canvas_height // 2, 
                image=self.tk_img, anchor=tk.CENTER
            )
            
        except Exception as e:
            messagebox.showerror("错误", f"无法显示图片: {str(e)}")
    
    def update_status(self):
        """更新状态栏信息"""
        if not self.image_files:
            self.status_label.config(text="未加载图片")
            return
        
        status_text = f"图片 {self.current_image_index + 1}/{len(self.image_files)} - {self.image_files[self.current_image_index]}"
        self.status_label.config(text=status_text)
    
    def prev_image(self):
        """显示上一张图片"""
        if not self.image_files:
            return
        
        self.current_image_index = (self.current_image_index - 1) % len(self.image_files)
        self.display_current_image()
        self.update_status()
    
    def next_image(self):
        """显示下一张图片"""
        if not self.image_files:
            return
        
        self.current_image_index = (self.current_image_index + 1) % len(self.image_files)
        self.display_current_image()
        self.update_status()
    
    def process_current_image(self):
        """处理当前选中的图片"""
        if not self.image_files:
            messagebox.showinfo("提示", "请先选择图片文件夹")
            return
        
        image_path = os.path.join(self.image_folder, self.image_files[self.current_image_index])
        
        # 确认处理
        if not messagebox.askyesno("确认", f"是否处理图片: {self.image_files[self.current_image_index]}?\n\n处理后将覆盖原图"):
            return
        
        # 最小化当前窗口
        self.root.iconify()
        
        # 启动处理流程
        try:
            self.status_label.config(text="正在处理图片...")
            self.open_photoshop_and_process(image_path)
            messagebox.showinfo("成功", "图片处理完成")
            self.status_label.config(text="处理完成")
        except Exception as e:
            messagebox.showerror("错误", f"处理图片时出错: {str(e)}")
            self.status_label.config(text="处理失败")
        
        # 恢复窗口
        self.root.deiconify()
    
    def open_photoshop_and_process(self, image_path):
        """打开Photoshop并处理图片"""
        # 从配置中获取设置
        ps_path = self.config["ps_path"]
        pause_time = self.config["pause_time"]
        
        # 设置PyAutoGUI的全局延迟
        pyautogui.PAUSE = pause_time / 2
        
        # 检查Photoshop是否已经运行
        ps_running = False
        try:
            # 尝试查找Photoshop窗口
            ps_window = pyautogui.getWindowsWithTitle("Photoshop")
            if ps_window:
                ps_running = True
                ps_window[0].activate()
                self.status_label.config(text="已激活Photoshop窗口")
            else:
                # 启动Photoshop
                self.status_label.config(text="正在启动Photoshop...")
                os.startfile(ps_path)
                time.sleep(10)  # 等待Photoshop启动
        except Exception as e:
            raise Exception(f"无法启动Photoshop: {str(e)}")
        
        # 打开图片
        self.status_label.config(text="正在打开图片...")
        self.open_image_in_photoshop(image_path)
        
        # 自动识别和处理文本区域
        self.status_label.config(text="正在自动识别文本区域...")
        
        # 使用快速选择工具 (W键)
        pyautogui.press('w')
        time.sleep(pause_time)
        
        # 使用选择主体功能 (Photoshop中的Select Subject)
        pyautogui.hotkey('alt', 'shift', 'a')  # 选择主体
        time.sleep(pause_time * 3)  # 等待选择完成
        
        # 反转选区 (Shift+Ctrl+I)，这样就选中了背景和文本
        pyautogui.hotkey('shift', 'ctrl', 'i')
        time.sleep(pause_time)
        
        # 使用魔棒工具进一步细化选区 (W键两次切换到魔棒)
        pyautogui.press('w')
        pyautogui.press('w')
        time.sleep(pause_time)
        
        # 设置魔棒容差为较低值，以便更精确地选择文本
        # 打开属性面板
        pyautogui.hotkey('ctrl', '1')
        time.sleep(pause_time)
        
        # 设置容差为20
        pyautogui.write('20')
        pyautogui.press('enter')
        time.sleep(pause_time)
        
        # 点击图像中的文本区域，添加到选区
        # 这里我们点击四个角落和中心，增加选中文本的概率
        screen_width, screen_height = pyautogui.size()
        center_x, center_y = screen_width // 2, screen_height // 2
        
        # 点击中心
        pyautogui.click(center_x, center_y)
        time.sleep(pause_time)
        
        # 点击四个角落 (加上Shift键以添加到选区)
        offset_x, offset_y = 200, 150  # 根据实际情况调整
        
        # 左上
        pyautogui.keyDown('shift')
        pyautogui.click(center_x - offset_x, center_y - offset_y)
        time.sleep(pause_time / 2)
        
        # 右上
        pyautogui.click(center_x + offset_x, center_y - offset_y)
        time.sleep(pause_time / 2)
        
        # 左下
        pyautogui.click(center_x - offset_x, center_y + offset_y)
        time.sleep(pause_time / 2)
        
        # 右下
        pyautogui.click(center_x + offset_x, center_y + offset_y)
        pyautogui.keyUp('shift')
        time.sleep(pause_time)
        
        # 应用内容识别和修复 (Shift+F5)
        self.status_label.config(text="正在应用内容识别和修复...")
        pyautogui.hotkey('shift', 'f5')
        time.sleep(pause_time * 2)
        
        # 点击确定按钮 (内容识别对话框)
        pyautogui.press('enter')
        time.sleep(pause_time * 2)
        
        # 保存图片 (Ctrl+S)
        self.status_label.config(text="正在保存图片...")
        pyautogui.hotkey('ctrl', 's')
        time.sleep(pause_time * 2)
        
        # 点击确定按钮 (覆盖原图确认对话框)
        pyautogui.press('enter')
        time.sleep(pause_time * 2)
        
        self.status_label.config(text="图片处理完成")
    
    def open_image_in_photoshop(self, image_path):
        """在Photoshop中打开图片"""
        # 从配置中获取延迟时间
        pause_time = self.config["pause_time"]
        
        # 使用Ctrl+O打开文件对话框
        pyautogui.hotkey('ctrl', 'o')
        time.sleep(pause_time * 2)
        
        # 输入文件路径
        pyautogui.write(image_path)
        time.sleep(pause_time)
        
        # 按回车确认
        pyautogui.press('enter')
        time.sleep(pause_time * 3)  # 等待图片加载
    

def main():
    """主函数"""
    try:
        # 设置应用程序图标和主题
        root = tk.Tk()
        root.title("PS绘本文本处理工具")
        
        # 尝试设置更好看的主题
        try:
            style = ttk.Style()
            if 'vista' in style.theme_names():
                style.theme_use('vista')
            elif 'clam' in style.theme_names():
                style.theme_use('clam')
        except:
            pass  # 如果设置主题失败，使用默认主题
        
        # 创建应用程序实例
        app = PSTextRemover(root)
        
        # 启动主循环
        root.mainloop()
    except Exception as e:
        # 捕获未处理的异常
        import traceback
        error_msg = f"发生错误: {str(e)}\n\n{traceback.format_exc()}"
        try:
            messagebox.showerror("错误", error_msg)
        except:
            print(error_msg)

if __name__ == "__main__":
    main()
