import os

from ultralytics import YOLO, SAM
from PIL import Image
import numpy as np

# 初始化模型
sam_model = SAM("./modelWeights/sam2.1_l.pt")
model_det_all = YOLO('./modelWeights/yolov8x-worldv2.pt')

def yoloWithSamMixedImage(task_type, detection_type, box_threshold, pilImage, strfileName, text_prompt=None):
    """
    处理输入的图片，根据给定的任务类型（检测或分割）和检测类型（基于文本或全检测），
    返回处理后的混合图像。

    :param task_type: 'seg' 表示分割任务，其他值表示检测任务
    :param detection_type: 'text' 基于文本检测，'all' 检测所有对象
    :param box_threshold: 置信度阈值
    :param pilImage: 图像数组
    :param text_prompt: 文本提示（当detection_type为'text'时有效）
    :return: 混合图像的PIL图像
    """
    pimg = pilImage
    img = np.array(pimg)

    if detection_type == 'text':
        model_det_all.set_classes(text_prompt.split(";"))
        results = model_det_all.predict(img, conf=box_threshold)
    else:
        results = model_det_all.predict(img, conf=box_threshold)

    result = results[0]

    if len(result.boxes) > 0 and task_type == 'seg':
        boxes = result.boxes.xyxy
        sam_results = sam_model(result.orig_img, bboxes=boxes, device='gpu')  # 使用CPU而不是CUDA
        processed_image = sam_results[0].plot()
    else:
        processed_image = img

    output_image = Image.fromarray(processed_image)
    base_name = os.path.splitext(os.path.basename(strfileName))[0]
    ex_name = os.path.splitext(os.path.basename(strfileName))[1]
    file_name = base_name + "_mix" + ex_name
    strDir = os.path.join(os.path.expanduser("./assets"), file_name)
    output_image.save(strDir)


    return file_name


def yoloWithSamMaskImage(task_type, detection_type, box_threshold, pilImage, strfileName, text_prompt=None):
    """
    处理输入的图片，根据给定的任务类型（检测或分割）和检测类型（基于文本或全检测），
    返回处理后的只包含掩码的图像。

    :param task_type: 'seg' 表示分割任务，其他值表示检测任务
    :param detection_type: 'text' 基于文本检测，'all' 检测所有对象
    :param box_threshold: 置信度阈值
    :param pilImage: 图像数组
    :param text_prompt: 文本提示（当detection_type为'text'时有效）
    :return: 只包含掩码的PIL图像
    """
    pimg = pilImage
    img = np.array(pimg)

    if detection_type == 'text':
        model_det_all.set_classes(text_prompt.split(";"))
        results = model_det_all.predict(img, conf=box_threshold)
    else:
        results = model_det_all.predict(img, conf=box_threshold)

    result = results[0]

    if len(result.boxes) > 0 and task_type == 'seg':
        boxes = result.boxes.xyxy
        sam_results = sam_model(result.orig_img, bboxes=boxes, device='gpu')  # 使用CPU而不是CUDA

        # 提取掩码并合并
        masks = sam_results[0].masks.data.cpu().numpy()
        combined_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for mask in masks:
            combined_mask[mask] = 255

        # 创建只包含掩码的图像
        processed_image = combined_mask
    else:
        processed_image = np.zeros_like(img, dtype=np.uint8)

    output_image = Image.fromarray(processed_image)
    base_name = os.path.splitext(os.path.basename(strfileName))[0]
    ex_name = os.path.splitext(os.path.basename(strfileName))[1]
    file_name = base_name + "_ori" + ex_name
    strDir = os.path.join(os.path.expanduser("./assets"), file_name)
    output_image.save(strDir)

    return file_name


# if __name__ == "__main__":
#     input_image_path = "./assets/car.jpeg"  # 替换为你的图像路径
#
#     # 获取混合图像
#     output_mixed_image = yoloWithSamMixedImage(task_type='seg', detection_type='text', box_threshold=0.25,
#                                                image_path=input_image_path, text_prompt="car")
#     output_mixed_image.show()  # 显示混合图像
#
#     # 获取只包含掩码的图像
#     output_mask_image = yoloWithSamMaskImage(task_type='seg', detection_type='text', box_threshold=0.25,
#                                              image_path=input_image_path, text_prompt="car")
#     output_mask_image.show()  # 显示只包含掩码的图像