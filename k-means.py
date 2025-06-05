import ee
import streamlit as st
import geemap.foliumap as geemap
import os
import time
from datetime import datetime

# 设置代理
geemap.set_proxy(port=7890)

# 初始化GEE
try:
    ee.Initialize()
except Exception as e:
    st.error("请先进行GEE认证！")
    st.stop()

# 页面配置
st.set_page_config(
    page_title="GEE影像K均值聚类分析",
    page_icon="🌍",
    layout="wide"
)

# 侧边栏标题和信息
st.sidebar.title("GEE影像K均值聚类分析")
st.sidebar.info(
    """
    这是一个基于Google Earth Engine和Geemap的影像K均值聚类分析工具。
    支持对Landsat影像进行无监督分类。
    """
)

# 主界面布局
col1, col2 = st.columns([4, 1])

# 创建地图
Map = geemap.Map(center=[30.5, 114.3], zoom=8)

# 初始化会话状态
if 'roi_coordinates' not in st.session_state:
    st.session_state.roi_coordinates = None

# 右侧控制面板
with col2:
    st.header("参数设置")
    
    # 聚类数量
    n_clusters = st.slider("聚类数量", 2, 20, 5)
    
    # 采样点数量
    n_samples = st.slider("采样点数量", 1000, 10000, 5000, step=1000)
    
    # 数据集选择
    dataset = st.selectbox(
        "选择数据集",
        ["Landsat 7", "Landsat 8", "Landsat 9"]
    )
    
    # 波段选择
    if dataset == "Landsat 7":
        bands = st.multiselect("选择波段", 
                               ["B1", "B2", "B3", "B4", "B5", "B7"],
                               default=["B1", "B2", "B3", "B4", "B5", "B7"])
    else:
        bands = st.multiselect("选择波段", 
                               ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
                               default=["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"])
    
    # 时间范围选择
    col_date1, col_date2 = st.columns(2)
    with col_date1:
        start_date = st.date_input("开始日期", datetime(2023, 1, 1))
    with col_date2:
        end_date = st.date_input("结束日期", datetime(2023, 12, 31))
    
    # 转换日期为字符串格式
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    # ROI设置
    roi_options = st.radio(
        "ROI选择方式",
        ["使用GEE资产", "手动输入坐标"]
    )
    
    if roi_options == "使用GEE资产":
        roi = st.text_input('GEE资产路径', 'users/yamiletsharon250/wuhan')
    else:
        roi_text = st.text_area("输入ROI坐标 (经度,纬度 格式，每行一个点)",
                                placeholder="114.0,30.0\n114.5,30.0\n114.5,30.5\n114.0,30.5")
        if roi_text:
            try:
                coords = []
                for line in roi_text.strip().split('\n'):
                    if line.strip():
                        lon, lat = map(float, line.strip().split(','))
                        coords.append([lon, lat])
                if len(coords) >= 3:
                    coords.append(coords[0])
                    st.session_state.roi_coordinates = coords
                    st.success("ROI坐标已更新")
            except Exception as e:
                st.error(f"坐标格式错误: {str(e)}")
    
    if st.button("开始聚类分析"):
        with st.spinner("正在进行聚类分析..."):
            try:
                # 获取ROI几何对象
                if roi_options == "手动输入坐标":
                    if not st.session_state.roi_coordinates:
                        st.warning("请先输入有效的ROI坐标")
                        st.stop()
                    roi_geometry = ee.Geometry.Polygon(st.session_state.roi_coordinates)
                else:
                    try:
                        roi_geometry = ee.FeatureCollection(roi)
                    except:
                        st.warning("无法加载GEE资产，请检查路径是否正确")
                        st.stop()
                
                # 加载影像
                if dataset == "Landsat 7":
                    image = ee.Image('LANDSAT/LE7_TOA_1YEAR/2001')
                elif dataset == "Landsat 8":
                    image = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')\
                        .filterDate(start_date_str, end_date_str)\
                        .filterBounds(roi_geometry)\
                        .median()
                else:  # Landsat 9
                    image = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')\
                        .filterDate(start_date_str, end_date_str)\
                        .filterBounds(roi_geometry)\
                        .median()
                
                # 选择波段并裁剪到ROI区域
                image = image.select(bands).clip(roi_geometry)
                
                # 创建训练数据集
                training = image.sample(**{
                    'region': roi_geometry,
                    'scale': 30,
                    'numPixels': n_samples
                })
                
                # 训练聚类器
                clusterer = ee.Clusterer.wekaKMeans(n_clusters).train(training)
                
                # 应用聚类并限制在ROI区域内
                result = image.cluster(clusterer).clip(roi_geometry)
                
                # 显示结果
                Map.addLayer(result.randomVisualizer(), {}, '聚类结果')
                Map.addLayer(roi_geometry, {'color': 'red'}, 'ROI')
                
                # 缩放到ROI区域
                Map.centerObject(roi_geometry, 8)
                
                st.success("聚类分析完成！")
                
            except Exception as e:
                st.error(f"聚类分析失败: {str(e)}")

# 左侧地图显示
with col1:
    Map.to_streamlit(height=850)

# 页脚信息
st.markdown("---")
st.markdown("© 2024 GEE影像K均值聚类分析工具 | 使用Google Earth Engine和Geemap构建")