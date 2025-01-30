import ee
import os
import geemap
import importlib.util
from .default_process import defaultProcess


def loadUserFunction(strModuleName):
    if strModuleName is None:
        # 如果条件为True，则使用默认的userProcess方法
        return defaultProcess
    else:
        # 如果条件为False，则加载userProcess方法
        module_name = 'userModule.' + strModuleName
        file_path = 'utils\\userModule\\' + strModuleName + '.py'

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return getattr(module, 'userProcess')




async def eeDownloader(config:dict):
    if config['Option']['UserImage'] is not None:
        return config['Option']['UserImage']
    strSensor = ee.ImageCollection(config['Option']['Sensor'])
    eeGeoJSON = geemap.geojson_to_ee(config['Geojson'])
    eeRoi = eeGeoJSON.geometry()

    functionProcess = loadUserFunction(config['Option']['UserModule'])

    collection = strSensor.filterDate(config['Option']['StartDate'], config['Option']['EndDate']) \
        .filter(ee.Filter.lt(config['Option']['Filter'][0], int(config['Option']['Filter'][1]))) \
        .filterBounds(eeGeoJSON) \
        .map(functionProcess) \
        .select(config['Option']['Bands'])

    eeComposite = collection.median().clip(eeRoi)
    strDir = os.path.join(os.path.expanduser("../Resource"), 'Tif')
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
    return {"Name": config['Option']['FileName'] + ".tif"}