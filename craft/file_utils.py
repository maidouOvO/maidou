# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import imgproc

# 获取文件列表
def get_files(img_dir):
    """
    获取目录中的图像文件列表
    
    Args:
        img_dir: 图像目录路径
        
    Returns:
        图像文件列表，扩展名列表，文件名列表
    """
    imgs, masks, xmls = [], [], []
    for root, _, files in os.walk(img_dir):
        for file in files:
            ext = os.path.splitext(file)[1]
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png':
                imgs.append(os.path.join(root, file))
            elif ext == '.mask' or ext == '.txt':
                masks.append(os.path.join(root, file))
            elif ext == '.xml':
                xmls.append(os.path.join(root, file))

    return imgs, masks, xmls

def saveResult(img_file, img, boxes, dirname='./result/', verticals=None, texts=None):
    """
    保存检测结果
    
    Args:
        img_file: 图像文件路径
        img: 图像数据
        boxes: 检测到的文本框
        dirname: 保存目录
        verticals: 垂直文本标记
        texts: 文本内容
    """
    # 确保目录存在
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    # 获取文件名
    filename, ext = os.path.splitext(os.path.basename(img_file))

    # 创建结果图像
    res = np.copy(img)

    # 绘制检测框
    for i, box in enumerate(boxes):
        poly = np.array(box).astype(np.int32).reshape((-1))
        poly = poly.reshape(-1, 2)
        cv2.polylines(res, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
        ptColor = (0, 255, 255)
        if verticals is not None:
            if verticals[i]:
                ptColor = (255, 0, 0)

        if texts is not None:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            cv2.putText(res, "{}".format(texts[i]), (poly[0][0]+1, poly[0][1]+1), font, font_scale, (0, 0, 0), thickness=1)
            cv2.putText(res, "{}".format(texts[i]), tuple(poly[0]), font, font_scale, (0, 255, 255), thickness=1)

    # 保存结果图像
    cv2.imwrite(dirname + filename + '_result' + ext, res)
