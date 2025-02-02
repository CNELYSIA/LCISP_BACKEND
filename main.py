from datetime import datetime
import os

from fastapi import FastAPI, UploadFile, File, HTTPException
import ee
from starlette.middleware.cors import CORSMiddleware

from utils.ee_downloader import eeDownloader
from fastapi.responses import StreamingResponse
from PIL import Image
import io

from utils.geo_utils import *

ee.Initialize()
app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:5173",
    "http://localhost:3000",
]

# noinspection PyTypeChecker
app.add_middleware(

    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

@app.post("/eeDownload")
async def ee_download(config: dict):
    UPLOAD_DIRECTORY = "./assets/"
    strPath = await eeDownloader(config)
    if config['Option']['UserImage'] is not None:
        return {"file": strPath}
    if config['Option']['Sensor'] == '谷歌地图瓦片':
        return {"file": strPath}
    imgPath = os.path.join(UPLOAD_DIRECTORY, strPath)
    obj_manager = ImageManager()
    obj_manager.readImg(imgPath)
    obj_manager.truncatedLinearStretch(dblPercentile=2)
    obj_manager.saveImg('./assets', obj_manager.dictConvertedImages, '.tif', formEE=True)
    return {"file": strPath}

@app.post("/uploadImage")
async def upload_image(file: UploadFile = File(...)):
    UPLOAD_DIRECTORY = "./assets/"
    # 检查文件是否存在
    if not file:
        raise HTTPException(status_code=400, detail="未找到上传的文件")

    # 检查文件类型是否为图片
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="仅支持图片上传")

    # 确保上传目录存在
    if not os.path.exists(UPLOAD_DIRECTORY):
        os.makedirs(UPLOAD_DIRECTORY)

    # 获取文件扩展名
    ext = os.path.splitext(file.filename)[1].lower()

    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    new_filename = f"{timestamp}{ext}"
    destination = os.path.join(UPLOAD_DIRECTORY, new_filename)

    try:
        # 将上传的文件写入到目标路径
        with open(destination, "wb") as image_file:
            content = await file.read()  # 异步读取上传的文件内容
            image_file.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {str(e)}")

    return {"info": f"文件 '{new_filename}' 已成功上传", "filename": new_filename}


@app.post("/uploadPyModule")
async def upload_module(file: UploadFile = File(...)):
    UPLOAD_DIRECTORY = "./utils/userModule/"
    # 检查文件是否存在
    if not file:
        raise HTTPException(status_code=400, detail="未找到上传的文件")

    # 获取文件扩展名并检查是否为 .py 文件
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    if ext != '.py':
        raise HTTPException(status_code=400, detail="仅支持 .py 文件上传")

    # 确保上传目录存在
    if not os.path.exists(UPLOAD_DIRECTORY):
        os.makedirs(UPLOAD_DIRECTORY)

    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    new_filename = f"{timestamp}{ext}"
    destination = os.path.join(UPLOAD_DIRECTORY, new_filename)

    try:
        # 将上传的文件写入到目标路径
        with open(destination, "wb") as script_file:
            content = await file.read()  # 异步读取上传的文件内容
            script_file.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {str(e)}")

    # 去掉扩展名的文件名
    base_name = os.path.splitext(new_filename)[0]

    return {"info": f"文件 '{new_filename}' 已成功上传", "filename": base_name}


@app.get("/getThumbnail/{filename}")  # 定义GET请求路由，接收图片文件名为参数
async def get_thumbnail(filename: str):
    try:
        # 构建图片路径（根据实际情况调整）
        img_path = f"./assets/{filename}"

        # 打开图片文件并生成缩略图
        with Image.open(img_path) as img:
            img.thumbnail((800, 800))  # 设置最大尺寸为800x800像素
            buffer = io.BytesIO()  # 创建一个内存中的字节缓冲区
            img.save(buffer, format="PNG", quality=95)  # 将缩略图保存到缓冲区，格式为JPEG，质量设为95%
            buffer.seek(0)  # 将缓冲区指针移到开头，以便读取
            # 返回StreamingResponse，直接将图片数据作为响应流返回
            return StreamingResponse(buffer, media_type="image/png")

    except Exception as e:
        # 如果发生任何异常，抛出HTTP 404错误，并返回错误详情
        raise HTTPException(status_code=404, detail=str(e))