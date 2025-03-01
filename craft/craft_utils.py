"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import numpy as np
import cv2
import math

""" 辅助函数 """
# 坐标变换
def warpCoord(Minv, pt):
    """
    变换坐标
    
    Args:
        Minv: 逆变换矩阵
        pt: 坐标点
        
    Returns:
        变换后的坐标
    """
    out = np.matmul(Minv, (pt[0], pt[1], 1))
    return np.array([out[0]/out[2], out[1]/out[2]])
""" 辅助函数结束 """


def getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text):
    """
    获取文本检测框的核心函数
    
    Args:
        textmap: 文本得分图
        linkmap: 链接得分图
        text_threshold: 文本置信度阈值
        link_threshold: 链接置信度阈值
        low_text: 文本低边界分数
        
    Returns:
        检测框，标签，映射
    """
    # 准备数据
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    """ 标记方法 """
    ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
    ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)

    text_score_comb = np.clip(text_score + link_score, 0, 1)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8), connectivity=4)

    det = []
    mapper = []
    for k in range(1,nLabels):
        # 大小过滤
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10: continue

        # 阈值处理
        if np.max(textmap[labels==k]) < text_threshold: continue

        # 创建分割图
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels==k] = 255
        segmap[np.logical_and(link_score==1, text_score==0)] = 0   # 移除链接区域
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        # 边界检查
        if sx < 0 : sx = 0
        if sy < 0 : sy = 0
        if ex >= img_w: ex = img_w
        if ey >= img_h: ey = img_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        # 创建框
        np_contours = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        # 对齐菱形
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = float(max(float(w), float(h))) / (float(min(float(w), float(h))) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = float(min(np_contours[:,0])), float(max(np_contours[:,0]))
            t, b = float(min(np_contours[:,1])), float(max(np_contours[:,1]))
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # 顺时针排序
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4-startidx, 0)
        box = np.array(box)

        det.append(box)
        mapper.append(k)

    return det, labels, mapper

def getPoly_core(boxes, labels, mapper, linkmap):
    """
    获取多边形的核心函数
    
    Args:
        boxes: 检测框
        labels: 标签
        mapper: 映射
        linkmap: 链接得分图
        
    Returns:
        多边形列表
    """
    # 配置
    num_cp = 5
    max_len_ratio = 0.7
    expand_ratio = 1.45
    max_r = 2.0
    step_r = 0.2

    polys = []  
    for k, box in enumerate(boxes):
        # 小实例的大小过滤
        w, h = int(np.linalg.norm(box[0] - box[1]) + 1), int(np.linalg.norm(box[1] - box[2]) + 1)
        if w < 10 or h < 10:
            polys.append(None); continue

        # 变换图像
        tar = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(box, tar)
        word_label = cv2.warpPerspective(labels, M, (w, h), flags=cv2.INTER_NEAREST)
        try:
            Minv = np.linalg.inv(M)
        except:
            polys.append(None); continue

        # 选定标签的二值化
        cur_label = mapper[k]
        word_label[word_label != cur_label] = 0
        word_label[word_label > 0] = 1

        """ 多边形生成 """
        # 查找顶部/底部轮廓
        cp = []
        max_len = -1
        for i in range(w):
            region = np.where(word_label[:,i] != 0)[0]
            if len(region) < 2 : continue
            cp.append((i, region[0], region[-1]))
            length = region[-1] - region[0] + 1
            if length > max_len: max_len = length

        # 如果max_len与h相似，则跳过
        if h * max_len_ratio < max_len:
            polys.append(None); continue

        # 获取固定长度的关键点
        tot_seg = num_cp * 2 + 1
        seg_w = w / tot_seg     # 段宽
        pp = [None] * num_cp    # 初始化关键点
        cp_section = [[0, 0]] * tot_seg
        seg_height = [0] * num_cp
        seg_num = 0
        num_sec = 0
        prev_h = -1
        for i in range(0,len(cp)):
            (x, sy, ey) = cp[i]
            if (seg_num + 1) * seg_w <= x and seg_num <= tot_seg:
                # 平均前一段
                if num_sec == 0: break
                cp_section[seg_num] = [float(cp_section[seg_num][0]) / num_sec, float(cp_section[seg_num][1]) / num_sec]
                num_sec = 0

                # 重置变量
                seg_num += 1
                prev_h = -1

            # 累积中心点
            cy = (sy + ey) * 0.5
            cur_h = ey - sy + 1
            cp_section[seg_num] = [cp_section[seg_num][0] + x, cp_section[seg_num][1] + cy]
            num_sec += 1

            if seg_num % 2 == 0: continue # 无多边形区域

            if prev_h < cur_h:
                idx = int((seg_num - 1)/2)
                pp[idx] = (x, cy)
                seg_height[idx] = cur_h
                prev_h = cur_h

        # 处理最后一段
        if num_sec != 0:
            cp_section[-1] = [float(cp_section[-1][0]) / num_sec, float(cp_section[-1][1]) / num_sec]

        # 如果关键点不足或段宽小于字符高度，则跳过
        if None in pp or seg_w < np.max(seg_height) * 0.25:
            polys.append(None); continue

        # 计算关键点的中值最大值
        half_char_h = np.median(seg_height) * expand_ratio / 2

        # 计算梯度并应用以创建水平关键点
        new_pp = []
        for i, (x, cy) in enumerate(pp):
            dx = cp_section[i * 2 + 2][0] - cp_section[i * 2][0]
            dy = cp_section[i * 2 + 2][1] - cp_section[i * 2][1]
            if dx == 0:     # 梯度为零
                new_pp.append([x, cy - half_char_h, x, cy + half_char_h])
                continue
            rad = - math.atan2(dy, dx)
            c, s = half_char_h * math.cos(rad), half_char_h * math.sin(rad)
            new_pp.append([x - s, cy - c, x + s, cy + c])

        # 获取边缘点以覆盖字符热图
        isSppFound, isEppFound = False, False
        grad_s = (pp[1][1] - pp[0][1]) / (pp[1][0] - pp[0][0]) + (pp[2][1] - pp[1][1]) / (pp[2][0] - pp[1][0])
        grad_e = (pp[-2][1] - pp[-1][1]) / (pp[-2][0] - pp[-1][0]) + (pp[-3][1] - pp[-2][1]) / (pp[-3][0] - pp[-2][0])
        for r in np.arange(0.5, max_r, step_r):
            dx = 2 * half_char_h * r
            if not isSppFound:
                line_img = np.zeros(word_label.shape, dtype=np.uint8)
                dy = grad_s * dx
                p = np.array(new_pp[0]) - np.array([dx, dy, dx, dy])
                cv2.line(line_img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), 1, thickness=1)
                if np.sum(np.logical_and(word_label, line_img)) == 0 or r + 2 * step_r >= max_r:
                    spp = p
                    isSppFound = True
            if not isEppFound:
                line_img = np.zeros(word_label.shape, dtype=np.uint8)
                dy = grad_e * dx
                p = np.array(new_pp[-1]) + np.array([dx, dy, dx, dy])
                cv2.line(line_img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), 1, thickness=1)
                if np.sum(np.logical_and(word_label, line_img)) == 0 or r + 2 * step_r >= max_r:
                    epp = p
                    isEppFound = True
            if isSppFound and isEppFound:
                break

        # 如果未找到多边形边界，则跳过
        if not (isSppFound and isEppFound):
            polys.append(None); continue

        # 创建最终多边形
        poly = []
        poly.append(warpCoord(Minv, (spp[0], spp[1])))
        for p in new_pp:
            poly.append(warpCoord(Minv, (p[0], p[1])))
        poly.append(warpCoord(Minv, (epp[0], epp[1])))
        poly.append(warpCoord(Minv, (epp[2], epp[3])))
        for p in reversed(new_pp):
            poly.append(warpCoord(Minv, (p[2], p[3])))
        poly.append(warpCoord(Minv, (spp[2], spp[3])))

        # 添加到最终结果
        polys.append(np.array(poly))

    return polys

def getDetBoxes(textmap, linkmap, text_threshold, link_threshold, low_text, poly=False):
    """
    获取检测框
    
    Args:
        textmap: 文本得分图
        linkmap: 链接得分图
        text_threshold: 文本置信度阈值
        link_threshold: 链接置信度阈值
        low_text: 文本低边界分数
        poly: 是否返回多边形
        
    Returns:
        检测框，多边形
    """
    boxes, labels, mapper = getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text)

    if poly:
        polys = getPoly_core(boxes, labels, mapper, linkmap)
    else:
        polys = [None] * len(boxes)

    return boxes, polys

def adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net = 2):
    """
    调整结果坐标
    
    Args:
        polys: 多边形列表
        ratio_w: 宽度比例
        ratio_h: 高度比例
        ratio_net: 网络比例
        
    Returns:
        调整后的多边形列表
    """
    if len(polys) > 0:
        polys = np.array(polys)
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    return polys
