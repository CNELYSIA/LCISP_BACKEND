import shutil
import subprocess
from datetime import datetime
import os
from pathlib import Path

import mysql

from train import train
import uvicorn
import translators as ts
from fastapi import FastAPI, UploadFile, File, HTTPException
import ee
from starlette.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

from utils.create_zip import createZip
from utils.ee_downloader import eeDownloader
from fastapi.responses import StreamingResponse
from PIL import Image
import io
import zipfile
from utils.geo_utils import *
from fastapi.responses import FileResponse
from utils.lang_segment_anything import *
from utils.yolo_segment_anything import *
from utils.statistics_download_img import get_image_info
from utils.statistics_mask import analyze_mask
from mysql.connector import pooling, Error
ee.Initialize()

db_config = {
    "host": "localhost",
    "user": "root",
    "passwd": "12345678",
    "database": "ISPSQL"
}

connection_pool = pooling.MySQLConnectionPool(pool_name="mypool", pool_size=5, **db_config)


def get_db_connection():
    return connection_pool.get_connection()

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

os.environ ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.on_event("shutdown")
async def shutdown_event():
    # 关闭连接池中的所有连接
    connection_pool.closeall()

@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

from fastapi import HTTPException, status

@app.get("/login")
async def login(username: str, password: str):
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM T_USER_INFO WHERE username=%s", (username,))
        user = cursor.fetchone()
        if not user or user['password'] != password:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
            )
        return {
            "message": "Logged in successfully",
            "role": user['role'],
            "username": username,
        }
    finally:
        cursor.close()
        connection.close()


@app.get("/register")  # 改为GET请求
async def register(username: str, password: str):
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    try:
        # 检查用户名是否已存在
        cursor.execute("SELECT * FROM T_USER_INFO WHERE username=%s", (username,))
        existing_user = cursor.fetchone()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )

        # 插入新用户
        cursor.execute(
            "INSERT INTO T_USER_INFO (username, password, role) VALUES (%s, %s, %s)",
            (username, password, "user")  # 默认角色为user
        )
        connection.commit()

        return {
            "message": "User registered successfully",
            "username": username
        }
    except mysql.connector.Error as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    finally:
        cursor.close()
        connection.close()
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
    current_date = datetime.now().strftime("%Y年%m月%d日 %H时%M分%S秒") 
  # 生成日期字符串
     # 新增数据库日志记录
    try:
        log_to_database(f"{current_date} 下载影像")  # 调用日志记录函数
    except Error as e:
        print(f"日志记录失败: {str(e)}")
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
    current_date = datetime.now().strftime("%Y年%m月%d日 %H时%M分%S秒") 
  # 生成日期字符串
     # 新增数据库日志记录
    try:
        log_to_database(f"{current_date} 上传影像")  # 调用日志记录函数
    except Error as e:
        print(f"日志记录失败: {str(e)}")

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
    current_date = datetime.now().strftime("%Y年%m月%d日 %H时%M分%S秒") 
  # 生成日期字符串
     # 新增数据库日志记录
    try:
        log_to_database(f"{current_date} 上传脚本文件")  # 调用日志记录函数
    except Error as e:
        print(f"日志记录失败: {str(e)}")

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



@app.post("/uploadPth")
async def upload_weights(file: UploadFile = File(...)):
    UPLOAD_DIRECTORY = "./assets/"
    # 检查文件是否存在
    if not file:
        raise HTTPException(status_code=400, detail="未找到上传的文件")

    # 获取文件扩展名并检查是否为 .pth 文件
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    if ext != '.pth':
        raise HTTPException(status_code=400, detail="仅支持 .pth 文件上传")

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
    current_date = datetime.now().strftime("%Y年%m月%d日 %H时%M分%S秒") 
  # 生成日期字符串
     # 新增数据库日志记录
    try:
        log_to_database(f"{current_date} 上传权重文件")  # 调用日志记录函数
    except Error as e:
        print(f"日志记录失败: {str(e)}")


    return {"info": f"文件 '{new_filename}' 已成功上传", "filename": new_filename}


@app.post("/predict")
async def predict(config: dict):
    IMAGE_DIRECTORY = "./assets/"
    strExtractType = ts.translate_text(config['Extraction'], from_language='zh', to_language='en')
    print(strExtractType)
    strPath = os.path.join(IMAGE_DIRECTORY, config['FileName'])
    pilImage = Image.open(strPath).convert("RGB")
    if config['ModelName'] == 'LangSAM':
        strOrigin, strMixture =  predictAndSave(pilImage, strExtractType, config['FileName'])
        return {"Mixture": strMixture, "Origin": strOrigin}
    elif config['ModelName'] == 'YoloSAM':
        strOrigin, strMixture = yoloWithSam("seg", "text", 0.25, pilImage, config['FileName'], strExtractType)
        return {"Mixture": strMixture, "Origin": strOrigin}

    strFilePath = Path(config['FileName'])
    strFileName = strFilePath.stem + ".zip"

    if config['Weights'] is None:
        zip = createZip('./utils/ModelTrainer', [strPath], config)
    else:
        pthPath = os.path.join(IMAGE_DIRECTORY, config['Weights'])
        Paths = [pthPath, strPath]
        zip = createZip('./utils/ModelTrainer', Paths, config)
    SAVE_DIRECTORY = "./temp/"
    strSavePath = os.path.join(SAVE_DIRECTORY, strFileName)
    RET_DIRECTORY = "./temp/" + Path(strFileName).stem
    RUN_DIRECTORY = "./temp/" + Path(strFileName).stem + '/predictor.py'
    with open(strSavePath, 'wb') as f:
        f.write(zip.getbuffer())
    zipFile = zipfile.ZipFile(strSavePath)
    zipFile.extractall(path=RET_DIRECTORY)
    subprocess.run(['python', 'predictor.py'], cwd=RET_DIRECTORY)
    current_date = datetime.now().strftime("%Y年%m月%d日 %H时%M分%S秒") 
  # 生成日期字符串
     # 新增数据库日志记录
    try:
        log_to_database(f"{current_date} 开始预测")  # 调用日志记录函数
    except Error as e:
        print(f"日志记录失败: {str(e)}")
    if strFilePath.suffix == '.tif':
        return {"Mixture": strFilePath.stem + '_mix.png', "Origin": strFilePath.stem + '_ori.tif'}
    else:
        return {"Mixture": strFilePath.stem + '_mix.png', "Origin": strFilePath.stem + '_ori.png'}
@app.get("/download/")
async def download_file(filename: str):
    IMAGE_DIRECTORY = "./assets/"
    file_path = os.path.join( IMAGE_DIRECTORY, filename)

    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="文件未找到")
    current_date = datetime.now().strftime("%Y年%m月%d日 %H时%M分%S秒") 
  # 生成日期字符串
     # 新增数据库日志记录
    try:
        log_to_database(f"{current_date} 下载影像")  # 调用日志记录函数
    except Error as e:
        print(f"日志记录失败: {str(e)}")
    # 使用FileResponse来发送文件
    return FileResponse(path=file_path, filename=filename, media_type='application/octet-stream')

@app.get("/getInfo/")
async def get_info(filename: str):
    IMAGE_DIRECTORY = "./assets/"
    file_path = os.path.join( IMAGE_DIRECTORY, filename)
    info = get_image_info(file_path)
    current_date = datetime.now().strftime("%Y年%m月%d日 %H时%M分%S秒") 
  # 生成日期字符串
     # 新增数据库日志记录
    try:
        log_to_database(f"{current_date} 获取影像信息")  # 调用日志记录函数
    except Error as e:
        print(f"日志记录失败: {str(e)}")
    return info

@app.get("/getMaskInfo/")
async def get_Mask_info(filename: str):
    IMAGE_DIRECTORY = "./assets/"
    file_path = os.path.join( IMAGE_DIRECTORY, filename)
    IMG = cv2.imread(file_path)
    info = analyze_mask(IMG)
    current_date = datetime.now().strftime("%Y年%m月%d日 %H时%M分%S秒") 
  # 生成日期字符串
     # 新增数据库日志记录
    try:
        log_to_database(f"{current_date} 获取掩膜要素信息")  # 调用日志记录函数
    except Error as e:
        print(f"日志记录失败: {str(e)}")
    return info

@app.post("/saveGeoTiff/")
async def save_GeoTiff(data: dict):
    IMAGE_DIRECTORY = "./assets/"
    originalFileName = data.get('originalFileName')
    oriFileName = data.get('oriFileName')

    if not originalFileName or not oriFileName:
        raise HTTPException(status_code=400, detail="文件名缺失")

    originalFilePath = os.path.join(IMAGE_DIRECTORY, originalFileName)
    oriFilePath = os.path.join(IMAGE_DIRECTORY, oriFileName)

    if not os.path.isfile(originalFilePath) or not os.path.isfile(oriFilePath):
        raise HTTPException(status_code=404, detail="文件未找到")

    obj_manager = ImageManager()
    obj_manager.readImg(originalFilePath)
    obj_manager.readImg(oriFilePath, append=True)

    obj_manager.assignGeoreference()


    # 保存带有地理信息的图像为 GeoTiff 文件
    savePath = os.path.join(IMAGE_DIRECTORY, f"{oriFileName}")
    obj_manager.saveImg('assets', obj_manager.dictGeoReferencedImages, '.tif', formEE=True)
    current_date = datetime.now().strftime("%Y年%m月%d日 %H时%M分%S秒") 
  # 生成日期字符串
     # 新增数据库日志记录
    try:
        log_to_database(f"{current_date} 保存TIF文件")  # 调用日志记录函数
    except Error as e:
        print(f"日志记录失败: {str(e)}")
    # 返回文件路径
    return FileResponse(path=savePath, filename=f"{oriFileName}", media_type='application/octet-stream')


@app.post("/saveShpFile/")
async def save_ShpFile(data: dict):
    filename = data.get('filename')
    IMAGE_DIRECTORY = "./assets/"
    filename_base = Path(filename).stem

    # 生成shp文件路径
    shp_dir = os.path.join(IMAGE_DIRECTORY, "temp_shp")
    os.makedirs(shp_dir, exist_ok=True)

    # 生成shp相关文件
    obj_manager = ImageManager()
    obj_manager.readImg(os.path.join(IMAGE_DIRECTORY, filename))
    obj_manager.tif2shp(shp_dir)  # 确保输出到临时目录

    # 创建ZIP文件
    zip_path = os.path.join(IMAGE_DIRECTORY, f"{filename_base}_shp.zip")
    shp_files = [f for f in os.listdir(shp_dir) if f.startswith(filename_base)]

    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file in shp_files:
            file_path = os.path.join(shp_dir, file)
            zipf.write(file_path, arcname=file)

    # 清理临时文件
    for file in shp_files:
        os.remove(os.path.join(shp_dir, file))
    os.rmdir(shp_dir)
    current_date = datetime.now().strftime("%Y年%m月%d日 %H时%M分%S秒") 
  # 生成日期字符串
     # 新增数据库日志记录
    try:
        log_to_database(f"{current_date} 保存SHP文件")  # 调用日志记录函数
    except Error as e:
        print(f"日志记录失败: {str(e)}")
    # 返回ZIP文件
    return FileResponse(
        path=zip_path,
        filename=f"{filename_base}_shp.zip",
        media_type='application/zip'
    )

def fileIter(path, format):
    files = []
    for file in os.listdir(path):
        full_path = os.path.join(path, file)
        if os.path.isdir(full_path):
            # 如果是子目录，递归调用函数
            files.extend(fileIter(full_path, format))
        elif file.endswith(format):
            # 添加到列表
            files.append(full_path)
    return files
@app.get("/update")
async def Update():
    pyScript = fileIter(r"../User", format=".py")
    logDirectories = []
    for root, dirs, files in os.walk(r"../User"):
        for subdir in dirs:
            logDirectories.append(os.path.join(root, subdir))
    List = {
        "logDirectories": logDirectories,
        "pyScript": pyScript
    }
    current_date = datetime.now().strftime("%Y年%m月%d日 %H时%M分%S秒") 
  # 生成日期字符串
     # 新增数据库日志记录
    try:
        log_to_database(f"{current_date} 搜索脚本文件")  # 调用日志记录函数
    except Error as e:
        print(f"日志记录失败: {str(e)}")
    return List
async def defaultTrainer(CONFIG):
    train(CONFIG)


async def userTrainer(file_path):
    oriPath = os.path.abspath('.')
    Path = os.path.abspath('..') + file_path[2:]
    pyPath = os.path.basename(Path)
    userPath = os.path.dirname(Path)
    os.chdir(userPath)
    subprocess.run(f'start cmd /K python {pyPath} -W {userPath}', shell=True)
    os.chdir(oriPath)


@app.post("/train")
async def Train(CONFIG: dict):
    if CONFIG['userType'] == '0':
        await defaultTrainer(CONFIG)
    elif CONFIG['userType'] == '1':
        await userTrainer(CONFIG['userPy'])
    current_date = datetime.now().strftime("%Y年%m月%d日 %H时%M分%S秒") 
  # 生成日期字符串
     # 新增数据库日志记录
    try:
        log_to_database(f"{current_date} 训练模型")  # 调用日志记录函数
    except Error as e:
        print(f"日志记录失败: {str(e)}")


# 添加知识图谱相关路径
KNOWLEDGE_GRAPH_DIR = "./LightRAG/documents"
os.makedirs(KNOWLEDGE_GRAPH_DIR, exist_ok=True)



import os
import subprocess

@app.post("/knowledgeGraph")
async def process_knowledge_graph(data: dict):

     # 固定文件名
    txt_path = os.path.join(KNOWLEDGE_GRAPH_DIR, "book.txt")

    # 保存文本内容
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(data["content"])

    dickens_dir = os.path.join("./LightRAG", "dickens")
    if os.path.exists(dickens_dir):
        shutil.rmtree(dickens_dir)
    # 执行 LightRAG/run_examples.bat 脚本
    run_script_path = os.path.join("./LightRAG", "run_examples.bat")
    if os.path.exists(run_script_path):
        try:
            subprocess.Popen(['cmd', '/K', 'run_examples.bat'], cwd=os.path.abspath("./LightRAG"),
                              creationflags=subprocess.CREATE_NEW_CONSOLE)
            print("Script executed successfully.")
        except Exception as e:
            print(f"Error executing script: {e}")
    else:
        print("Script does not exist.")
    current_date = datetime.now().strftime("%Y年%m月%d日 %H时%M分%S秒") 
  # 生成日期字符串
     # 新增数据库日志记录
    try:
        log_to_database(f"{current_date} 生成知识图谱")  # 调用日志记录函数
    except Error as e:
        print(f"日志记录失败: {str(e)}")
    return {"htmlPath": f"/LightRAG/knowledge_graph.html"}

from fastapi.staticfiles import StaticFiles

# 挂载静态文件目录
app.mount("/LightRAG", StaticFiles(directory="./LightRAG"), name="LightRAG")

from fastapi import status

def log_to_database(content: str):
    connection = get_db_connection()
    cursor = connection.cursor()
    try:
        cursor.execute(
            "INSERT INTO T_LOGS_INFO (logs) VALUES (%s)",
            (content,)
        )
        connection.commit()
        return cursor.lastrowid
    except Error as e:
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()

@app.get("/addLog")
async def add_log(log_content: str):
    connection = get_db_connection()
    cursor = connection.cursor()
    try:
        # 执行插入操作
        cursor.execute(
            "INSERT INTO T_LOGS_INFO (logs) VALUES (%s)",
            (log_content,)
        )
        connection.commit()

        # 获取刚插入的ID
        log_id = cursor.lastrowid

        return {
            "message": "日志写入成功",
            "log_id": log_id,
            "content": log_content
        }
    except Error as e:
        connection.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"数据库错误: {str(e)}"
        )
    finally:
        cursor.close()
        connection.close()


@app.get("/getlogs")
async def get_logs():
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    try:
        # 执行查询并按时间倒序排列
        cursor.execute("SELECT logs FROM T_LOGS_INFO")
        logs = cursor.fetchall()

        # 直接提取logs字段构建列表
        log_list = [log['logs'] for log in logs]

        return {"logs": log_list}

    except Error as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"数据库查询失败: {str(e)}"
        )
    finally:
        cursor.close()
        connection.close()