import os
import numpy as np
from PIL import Image
from lang_sam.lang_sam import LangSAM
import cv2
from lang_sam.utils import draw_image

def predictAndSave(pilImage, strPrompt, strfileName):
    model = LangSAM()
    results = model.predict([pilImage], [strPrompt])[0]

    # Calculate the maximum value across all bands to get a 2D array
    meanImage = np.max(results["masks"], axis=0)
    mean_image_8bit = cv2.normalize(meanImage, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Save the mask image
    base_name = os.path.splitext(os.path.basename(strfileName))[0]
    ex_name = os.path.splitext(os.path.basename(strfileName))[1]
    mask_file_name = base_name + "_ori" + ex_name
    mask_strDir = os.path.join(os.path.expanduser("./assets"), mask_file_name)
    cv2.imwrite(mask_strDir, mean_image_8bit)

    # Draw the origin image with masks
    image_array = np.asarray(pilImage)
    output_image = draw_image(
        image_array,
        results["masks"],
        results["boxes"],
        results["scores"],
        results["labels"],
    )
    output_image = Image.fromarray(np.uint8(output_image)).convert("RGB")

    # Save the origin image with masks
    mix_file_name = base_name + "_mix" + ex_name
    mix_strDir = os.path.join(os.path.expanduser("./assets"), mix_file_name)
    output_image.save(mix_strDir)

    return mask_file_name, mix_file_name

# # 示例调用
# pilImage = Image.open("P1027100.JPG")
# strPrompt = "flower"
# strfileName = "example.jpg"
# mask_file_name, mix_file_name = predictAndSave(pilImage, strPrompt, strfileName)
# print(f"Mask file saved as: {mask_file_name}")
# print(f"Mixed file saved as: {mix_file_name}")



