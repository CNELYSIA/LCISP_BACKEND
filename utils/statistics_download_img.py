import os
import cv2
from osgeo import gdal, ogr, osr


def get_image_info(image_path):
    # 获取文件扩展名
    _, ext = os.path.splitext(image_path)
    ext = ext.lower()

    if ext == '.tif':
        return get_tif_info(image_path)
    else:
        return get_opencv_info(image_path)


def get_opencv_info(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("无法读取图像文件")

    height, width, channels = image.shape
    info = {
        'height': height,
        'width': width,
        'resolution': str(width) + ' x ' + str(height),
        'bands': channels,
        'dtype': image.dtype.name,
        'driver': "无",
        'proj': "此文件不存在投影信息 / 正常示例：GEOGCS['WGS 84',DATUM['WGS_1984',SPHEROID['WGS 84',6378137,298.257223563,AUTHORITY['EPSG','7030']],AUTHORITY['EPSG','6326']],PRIMEM['Greenwich',0,AUTHORITY['EPSG','8901']],UNIT['degree',0.0174532925199433,AUTHORITY['EPSG','9122']],AXIS['Latitude',NORTH],AXIS['Longitude',EAST],AUTHORITY['EPSG','4326']]",
        'trans': "不存在几何变换参数 / 正常示例：[ 117.17124938964844, 0.000001341104507446289, 0, 36.68796847703968, 0, -0.0000010755487046205463 ]",
        'area': "缺少空间参考，无法计算"
    }
    return info


def get_tif_info(image_path):
    dataset = gdal.Open(image_path)
    if dataset is None:
        raise ValueError("无法读取TIF文件")

    width = dataset.RasterXSize
    height = dataset.RasterYSize
    bands = dataset.RasterCount
    dtype = gdal.GetDataTypeName(dataset.GetRasterBand(1).DataType)
    driver = dataset.GetDriver().LongName
    projection_ref = dataset.GetProjectionRef()
    geotransform = dataset.GetGeoTransform()

    area_meters = calculate_area(width, height, geotransform, projection_ref)

    info = {
        'height': height,
        'width': width,
        'resolution': str(width) + ' x ' + str(height),
        'bands': bands,
        'dtype': dtype,
        'driver': driver,
        'proj': projection_ref,
        'trans': geotransform,
        'area': area_meters
    }

    return info


def calculate_area(width, height, geotransform, projection_ref):
    def pixel_to_geo(x, y):
        x_origin, pixel_width, _, y_origin, _, pixel_height = geotransform
        lon = x_origin + x * pixel_width
        lat = y_origin + y * pixel_height
        return lat, lon

    ring = ogr.Geometry(ogr.wkbLinearRing)
    for point in [(0, 0), (width, 0), (width, height), (0, height), (0, 0)]:
        lat, lon = pixel_to_geo(point[0], point[1])
        ring.AddPoint(lon, lat)  # 注意顺序是lon,lat

    # 创建多边形并计算面积
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromWkt(projection_ref)
    poly.AssignSpatialReference(spatial_ref)

    # 使用原始坐标系计算面积
    area_meters_squared = poly.GetArea()

    return area_meters_squared
