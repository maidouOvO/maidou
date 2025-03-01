import numpy as np
import cv2
import math

def getDetBoxes(textmap, linkmap, text_threshold, link_threshold, low_text, poly=False):
    """获取文本检测框"""
    # 准备数据
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    # 阈值处理
    _, text_score = cv2.threshold(textmap, low_text, 1, 0)
    _, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)

    # 合并文本得分和链接得分
    text_score_comb = np.clip(text_score + link_score, 0, 1)

    # 查找连通区域
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8), connectivity=4)

    # 准备输出数组
    det = []
    mapper = []
    
    # 对每个连通区域进行处理
    for k in range(1, nLabels):
        # 获取连通区域的大小和得分
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10: continue

        # 获取连通区域的掩码
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels==k] = 255
        segmap[np.logical_and(link_score==1, text_score==0)] = 0  # 移除链接区域
        
        # 计算连通区域的得分
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        # 边界检查
        if sx < 0: sx = 0
        if sy < 0: sy = 0
        if ex >= img_w: ex = img_w
        if ey >= img_h: ey = img_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        # 确保最小得分大于阈值
        np_contours = np.roll(np.array(np.where(segmap!=0)), 1, axis=0).transpose().reshape(-1, 2)
        if len(np_contours) == 0:
            continue
            
        # 计算多边形或矩形框
        if poly:
            # 获取多边形框
            hull = cv2.convexHull(np_contours).reshape(-1, 2)
            if len(hull) < 4:
                continue
            # 近似多边形
            approx_curve = cv2.approxPolyDP(hull, 3, True)
            approx = approx_curve.reshape(-1, 2)
            if len(approx) < 4:
                continue
            # 确保多边形是凸的
            approx = np.array(approx)
            if not cv2.isContourConvex(approx):
                continue
        else:
            # 获取矩形框
            rect = cv2.minAreaRect(np_contours)
            box = cv2.boxPoints(rect)
            approx = box
            
        # 计算框的得分
        approx_area = cv2.contourArea(approx)
        if approx_area < 5:
            continue
            
        # 找到框内的所有像素
        tmp = np.zeros(segmap.shape, dtype=np.uint8)
        cv2.fillPoly(tmp, [approx.astype(np.int32)], 1)
        mask = np.logical_and(tmp, segmap)
        score = np.sum(textmap[mask]) / np.sum(mask)
        
        # 如果得分低于阈值，跳过
        if score < text_threshold:
            continue
            
        # 输出框和得分
        det.append(approx)
        mapper.append(k)
        
    return det, labels, mapper

def adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net=2):
    """调整检测结果的坐标"""
    if len(polys) > 0:
        polys = np.array(polys)
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    return polys
