from datetime import datetime
import os

from fastapi import FastAPI, UploadFile, File, HTTPException
import ee
from starlette.middleware.cors import CORSMiddleware

from utils.ee_downloader import eeDownloader
# 测试时注释掉
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
   strPath = await eeDownloader(config)
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

    # 去掉扩展名的文件名
    base_name = os.path.splitext(new_filename)[0]

    return {"info": f"文件 '{new_filename}' 已成功上传", "filename": base_name}


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