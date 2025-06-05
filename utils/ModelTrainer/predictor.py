import json
import os
import shutil
from pathlib import Path
from shutil import copy2

import cv2
import nest_asyncio
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from models import SGCNNet
from models import UNet
from models import NlLinkNet
import pandas as pd
from utils import geo_utils
from utils.geo_utils import ImageManager
import asyncio
import aiofiles
nest_asyncio.apply()
dataset_Rootdir = 'Crops'


async def predict_single_image(config, image_name, num_classes):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.2304, 0.3295, 0.4405], std=[0.1389, 0.1316, 0.1278])
        ]
    )

    root_path = dataset_Rootdir
    image_path = os.path.join(root_path, image_name)

    image = Image.open(image_path)
    image = transform(image).float().cuda()
    # batch_size=1
    image = image.unsqueeze(0)

    model = load_model(config)
    output = model(image)
    _, pred = output.max(1)
    pred = pred.view(config['img_width'], config['img_height'])
    mask_im = pred.cpu().numpy().astype(np.uint8)

    pre_base_path = config['pre_dir']
    pre_mask_path = os.path.join(pre_base_path, 'mask')
    pre_vis_path = os.path.join(pre_base_path, 'vis')

    file_name = image_name.split('\\')[-1]
    save_label = os.path.join(pre_mask_path, file_name)
    await write_image_async(save_label, mask_im)
    print("写入{}成功".format(save_label))
    save_visual = os.path.join(pre_vis_path, file_name)
    print("开始写入{}".format(save_visual))
    translabeltovisual(save_label, save_visual, num_classes)
    print("写入{}成功".format(save_visual))


def load_model(config):
    device = torch.device('cuda:0')
    selected = config['predict_model']['model'][config['predict_model']['select']]
    if selected == 'SGCNNet':
        model = SGCNNet.SGCN_res50(num_classes=config['num_classes'])
    elif selected == 'LinkNet':
        model = NlLinkNet.NL34_LinkNet()
    elif selected == 'UNet':
        model = UNet.UNET()

    if config['userWeights']:
        check_point =  os.path.join(config['save_model']['save_path'],config['weights'])
    else:
        check_point = os.path.join(config['save_model']['save_path'],
                                   selected + '_' + config['extraction_type'] + '.pth')
    print("加载模型{}成功".format(check_point))
    model.load_state_dict(torch.load(check_point), False)
    model.cuda()
    model.eval()
    return model


def translabeltovisual(save_label, path, num_classes):
    im = cv2.imread(save_label)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            pred_class = im[i][j][0]
            im[i][j] = num_classes[pred_class]
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, im)


async def write_image_async(file_path, image_array):
    with open(file_path, 'wb') as f:
        ret, buffer = cv2.imencode('.png', image_array)
        f.write(buffer.tobytes())


with open('config.json', 'r', encoding='utf-8') as f:
    df = json.load(f)

Manager = ImageManager()

Manager.readImg(df['FileName'])
Manager.cropImg(intWidth=512, intHeight=512, intStep=256, intStartGroup=1)
Manager.saveImg('./Crops', Manager.dictCroppedImages, '.png', formEE=False)

with open('Predict.txt', 'w') as f:
    for key in Manager.dictCroppedImages.keys():
        f.writelines(key + '.png\n')

# 定义一个字符串映射字典
MapPing = {
    "道路": "road",
    "耕地": "cul",
}

if df['Weights'] is None:
    PredictConfig = {
        "num_classes": 2,
        "pre_dir": "Crops",
        "img_txt": "Predict.txt",
        "img_height": 512,
        "img_width": 512,
        "predict_model": {
            "select": 0,
            "model": [df['ModelName']]
        },
        "userWeights": False,
        "extraction_type": MapPing[df['Extraction']],
        "save_model": {
            "save": "true",
            "save_path": "./defaultModelWeights"
        }
    }
elif df['Weights'] is not None:
    PredictConfig = {
        "num_classes": 2,
        "pre_dir": "Crops",
        "img_txt": "Predict.txt",
        "img_height": 512,
        "img_width": 512,
        "predict_model": {
            "select": 0,
            "model": [df['ModelName']]
        },
        "extraction_type": MapPing[df['Extraction']],
        "userWeights": True,
        "weights": df['Weights'],
        "save_model": {
            "save": "true",
            "save_path": "./"
        }
    }


async def predict(config, num_classes):
    pre_base_path = config['pre_dir']
    pre_mask_path = os.path.join(pre_base_path, 'mask')
    pre_vis_path = os.path.join(pre_base_path, 'vis')
    if os.path.exists(pre_base_path) is False:
        os.makedirs(pre_base_path)
    if os.path.exists(pre_mask_path) is False:
        os.makedirs(pre_mask_path)
    if os.path.exists(pre_vis_path) is False:
        os.makedirs(pre_vis_path)

    async with aiofiles.open(config['img_txt'], 'r', encoding='utf-8') as f:
        images = await f.readlines()

    tasks = []
    for image in images:
        task = asyncio.create_task(predict_single_image(config, image.strip(), num_classes))
        tasks.append(task)
    await asyncio.gather(*tasks)


asyncio.run(predict(PredictConfig, [[0, 0, 0], [255, 255, 255]]))

obj_Manager = ImageManager()
obj_Manager.readImg('./Crops/vis')
Manager.dictCroppedImages = obj_Manager.dictImages
Manager.stitchImg(intWidth=512, intHeight=512, intStep=256)
if df['FileName'].lower().endswith(('.tif', '.tiff')):
    Manager.savePredicted('./Predicted', Manager.dictStitchedImages, '.tif', formEE=False)
    strMaskName = Path(df['FileName']).stem + '_ori.tif'
else:
    Manager.savePredicted('./Predicted', Manager.dictStitchedImages, '.png', formEE=False)
    strMaskName = Path(df['FileName']).stem + '_ori.png'
print('Predict Success!')

strMaskPath = './Predicted/' + strMaskName
strSavePath ='./Predicted/' + Path(df['FileName']).stem + '_mix.png'
Mask = cv2.imread(strMaskPath)
Origin = cv2.imread(df['FileName'])
combine = cv2.addWeighted(Origin,0.5,Mask,0.5,0)
cv2.imwrite(strSavePath,combine)

shutil.copy2(strMaskPath,'../../assets/')
shutil.copy2(strSavePath,'../../assets/')