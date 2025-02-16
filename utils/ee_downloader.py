import ee
import os
import geemap
import importlib.util

import nest_asyncio

from .default_process import defaultProcess
from .geo_utils import fetchSatelliteDataReturnFileName
nest_asyncio.apply()

import ast

def is_safe_code(source):
    """
    进行简单的静态分析，检查代码中是否包含任何 import 语句。
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            return False
    return True

def loadUserFunction(strModuleName, defaultProcess):
    if strModuleName is None:
        # 如果条件为True，则使用默认的userProcess方法
        return defaultProcess
    else:
        # 如果条件为False，则加载userProcess方法
        module_name = 'userModule.' + strModuleName
        file_path = 'utils/userModule/' + strModuleName + '.py'

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
        except UnicodeDecodeError:
            raise ValueError(f"请使用UTF-8编码的Python文件.")

        if not is_safe_code(source_code):
            raise ValueError("存在不允许的引用.")

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return getattr(module, 'userProcess')


async def eeDownloader(config:dict):
    if config['Option']['UserImage'] is not None:
        return config['Option']['UserImage']
    if config['Option']['Sensor'] == '谷歌地图瓦片':
        return fetchSatelliteDataReturnFileName(
            intZoomLevel=int(20),
            strRootDirectory=os.path.expanduser("./assets"),
            geojsonData=config['Geojson'],
            strFileName=config['Option']['FileName']
        )
    strSensor = ee.ImageCollection(config['Option']['Sensor'])
    eeGeoJSON = geemap.geojson_to_ee(config['Geojson'])
    eeRoi = eeGeoJSON.geometry()

    functionProcess = loadUserFunction(config['Option']['UserModule'], defaultProcess)

    collection = strSensor.filterDate(config['Option']['StartDate'], config['Option']['EndDate']) \
        .filter(ee.Filter.lt(config['Option']['Filter'][0], int(config['Option']['Filter'][1]))) \
        .filterBounds(eeGeoJSON) \
        .map(functionProcess) \
        .select(config['Option']['Bands'])

    eeComposite = collection.median().clip(eeRoi)
    strDir = os.path.expanduser("./assets")
    if not os.path.exists(strDir):
        os.makedirs(strDir)
    strPath= os.path.join(strDir, config['Option']['FileName'] + ".tif")
    geemap.download_ee_image(
        image=eeComposite,
        filename=strPath,
        region=eeRoi,
        crs=config['Option']['Crs'],
        scale=int(config['Option']['Scale']),
    )
    return config['Option']['FileName'] + ".tif"