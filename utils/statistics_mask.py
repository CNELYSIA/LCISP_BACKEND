import cv2
import numpy as np
import json


def analyze_mask(mask):
    # 确保掩码是二值图像（黑白）
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 计算掩码面积
    area = cv2.countNonZero(binary_mask)
    # 获取图像的整体面积
    overall_area = mask.shape[0] * mask.shape[1]

    # 面积占比
    area_ratio = area / overall_area
    # 找到轮廓
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        main_contour = max(contours, key=cv2.contourArea)

        # 计算边界框
        x, y, w, h = cv2.boundingRect(main_contour)
        bounding_box_area = w * h

        # 计算质心
        M = cv2.moments(main_contour)
        centroid_x = int(M["m10"] / M["m00"])
        centroid_y = int(M["m01"] / M["m00"])

        # 计算周长
        perimeter = cv2.arcLength(main_contour, True)

        # 计算圆形度
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0

        # 计算凸包
        hull = cv2.convexHull(main_contour)
        convex_hull_area = cv2.contourArea(hull)

        # 计算连通域数量
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

        # 计算填充率
        fill_factor = area / bounding_box_area if bounding_box_area != 0 else 0

        # 计算惯性矩
        mu = cv2.moments(main_contour)
        central_mu_02 = mu['mu02'] / mu['m00']
        central_mu_20 = mu['mu20'] / mu['m00']
        central_mu_11 = mu['mu11'] / mu['m00']

        # 计算方向角
        orientation_angle = 0.5 * np.arctan2(2 * central_mu_11, central_mu_20 - central_mu_02) * (180 / np.pi)

        return {
            "area": area,
            "overallArea": overall_area,
            "areaRatio": area_ratio,
            "boundingBox": {"x": x, "y": y, "width": w, "height": h},
            "centroid": {"x": centroid_x, "y": centroid_y},
            "perimeter": perimeter,
            "circularity": circularity,
            "convexHull": hull.tolist(),
            "numConnectedComponents": num_labels - 1,  # 减去背景标签
            "fillFactor": fill_factor,
            "momentSofinertia": {"mu02": central_mu_02, "mu20": central_mu_20, "mu11": central_mu_11},
            "orientationAngle": orientation_angle
        }
    else:
        return {
            "area": 0,
            "overallArea": 100,
            "areaRatio": 0,
            "perimeter": 0,
            "circularity": 0,
            "convexHull": [[[0, 0]]],
            "numConnectedComponents": 0,  # 减去背景标签
            "fillFactor":0,
            "orientationAngle": 0
        }


