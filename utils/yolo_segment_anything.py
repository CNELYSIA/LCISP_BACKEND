import os

import cv2
from ultralytics import YOLO, SAM
from PIL import Image
import numpy as np

# 初始化模型
sam_model = SAM("./modelWeights/sam2.1_l.pt")
model_det_all = YOLO('./modelWeights/yolov8x-worldv2.pt')

def yoloWithSam(task_type, detection_type, box_threshold, pilImage, strfileName, text_prompt=None):
    """
    处理输入的图片，根据给定的任务类型（检测或分割）和检测类型（基于文本或全检测），
    返回处理后的混合图像和只包含掩码的图像。

    :param task_type: 'seg' 表示分割任务，其他值表示检测任务
    :param detection_type: 'text' 基于文本检测，'all' 检测所有对象
    :param box_threshold: 置信度阈值
    :param pilImage: 图像数组
    :param text_prompt: 文本提示（当detection_type为'text'时有效）
    :return: 混合图像的文件名和只包含掩码的图像的文件名
    """
    pimg = pilImage
    img = np.array(pimg).astype(np.float32)  # 转换为 float32 类型

    if detection_type == 'text':
        model_det_all.set_classes(text_prompt.split(";"))
        results = model_det_all.predict(img, conf=box_threshold)
    else:
        results = model_det_all.predict(img, conf=box_threshold)

    result = results[0]

    if len(result.boxes) > 0 and task_type == 'seg':
        boxes = result.boxes.xyxy
        sam_results = sam_model(result.orig_img.astype(np.float32), bboxes=boxes, device='cpu')  # 使用CPU而不是GPU

        # 提取掩码并合并
        masks = sam_results[0].masks.data.cpu().numpy()
        combined_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for mask in masks:
            combined_mask[mask] = 255

        # 创建只包含掩码的图像
        processed_mask_image = combined_mask

        # 绘制混合图像
        processed_mixed_image = sam_results[0].plot()
    else:
        processed_mask_image = np.zeros_like(img, dtype=np.uint8)
        processed_mixed_image = img

    # 保存只包含掩码的图像
    base_name = os.path.splitext(os.path.basename(strfileName))[0]
    ex_name = os.path.splitext(os.path.basename(strfileName))[1]
    mask_file_name = base_name + "_ori" + ex_name
    mask_strDir = os.path.join(os.path.expanduser("./assets"), mask_file_name)
    cv2.imwrite(mask_strDir, processed_mask_image)

    # 保存混合图像
    mixed_file_name = base_name + "_mix" + ex_name
    mixed_strDir = os.path.join(os.path.expanduser("./assets"), mixed_file_name)
    output_image = Image.fromarray(processed_mixed_image.astype(np.uint8))
    output_image.save(mixed_strDir)

    return mask_file_name, mixed_file_name

# 示例调用
# pilImage = Image.open("P1027100.JPG")
# strPrompt = "flower"
# strfileName = "example.jpg"
# mask_file_name, mixed_file_name = yoloWithSam('seg', 'text', 0.5, pilImage, strfileName, text_prompt=strPrompt)
# print(f"Mask file saved as: {mask_file_name}")
# print(f"Mixed file saved as: {mixed_file_name}")
#


