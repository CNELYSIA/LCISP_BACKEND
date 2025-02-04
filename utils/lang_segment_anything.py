import os
import numpy as np
from PIL import Image
from .lang_sam.lang_sam import LangSAM
import cv2

from .lang_sam.utils import draw_image


def predictWithMask(pilImage, strPrompt, strfileName):
    model = LangSAM()
    results = model.predict([pilImage], [strPrompt])[0]
    # 计算所有波段的平均值，得到一个二维数组
    meanImage = np.max(results["masks"], axis=0)
    mean_image_8bit = cv2.normalize(meanImage, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    base_name = os.path.splitext(os.path.basename(strfileName))[0]
    ex_name = os.path.splitext(os.path.basename(strfileName))[1]
    file_name = base_name + "_ori" + ex_name
    strDir= os.path.join(os.path.expanduser("./assets"),file_name)
    with open(strDir, 'wb') as f:
        cv2.imwrite(strDir, mean_image_8bit)
    return file_name

def predictWithOrigin(pilImage, strPrompt, strfileName):
    model = LangSAM()
    results = model.predict([pilImage], [strPrompt])[0]
    image_array = np.asarray(pilImage)
    output_image = draw_image(
        image_array,
        results["masks"],
        results["boxes"],
        results["scores"],
        results["labels"],
    )
    output_image = Image.fromarray(np.uint8(output_image)).convert("RGB")
    base_name = os.path.splitext(os.path.basename(strfileName))[0]
    ex_name = os.path.splitext(os.path.basename(strfileName))[1]
    file_name = base_name + "_mix" + ex_name
    strDir = os.path.join(os.path.expanduser("./assets"), file_name)
    output_image.save(strDir)
    return file_name