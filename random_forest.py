import ee
import streamlit as st
import geemap.foliumap as geemap
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
    page_title="GEE影像随机森林分类器",
    page_icon="🌍",
    layout="wide"
)

# 侧边栏标题和信息
st.sidebar.title("GEE影像随机森林分类器")
st.sidebar.info(
    """
    这是一个基于Google Earth Engine和Geemap的影像随机森林分类工具。
    支持对Landsat影像进行监督分类。
    """
)

# 主界面布局
col1, col2 = st.columns([4, 1])

# 创建地图
Map = geemap.Map(center=[40, -100], zoom=4)

# 右侧控制面板
with col2:
    st.header("分类参数")
    
    # 数据集选择
    dataset = st.selectbox(
        "选择数据集",
        ["Landsat 8"]
    )
    
    # 时间范围选择
    col_date1, col_date2 = st.columns(2)
    with col_date1:
        start_date = st.date_input("开始日期", datetime(2018, 1, 1))
    with col_date2:
        end_date = st.date_input("结束日期", datetime(2018, 12, 31))
    
    # 转换日期为字符串格式
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    # 波段选择
    bands = st.multiselect(
        "选择波段",
        ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', 'B11'],
        default=['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', 'B11']
    )
    
    # 训练样本选择
    training_samples = st.text_input(
        "训练样本路径",
        "GOOGLE/EE/DEMOS/demo_landcover_labels"
    )
    
    # 标签属性选择
    label_property = st.text_input(
        "标签属性名称",
        "landcover"
    )
    
    # 随机森林参数
    st.subheader("随机森林参数")
    n_trees = st.slider("决策树数量", 10, 100, 50)
    min_leaf_population = st.slider("最小叶子节点样本数", 1, 10, 1)
    
    if st.button("开始分类"):
        with st.spinner("正在进行随机森林分类..."):
            try:
                # 加载Landsat 8影像集
                l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1')
                
                # 创建合成影像
                image = ee.Algorithms.Landsat.simpleComposite(**{
                    'collection': l8.filterDate(start_date_str, end_date_str),
                    'asFloat': True
                })
                
                # 加载训练样本
                points = ee.FeatureCollection(training_samples)
                
                # 创建训练数据集
                training = image.select(bands).sampleRegions(**{
                    'collection': points,
                    'properties': [label_property],
                    'scale': 30
                })
                
                # 训练随机森林分类器
                trained = ee.Classifier.smileRandomForest(**{
                    'numberOfTrees': n_trees,
                    'minLeafPopulation': min_leaf_population
                }).train(training, label_property, bands)
                
                # 应用分类
                classified = image.select(bands).classify(trained)
                
                # 显示结果
                Map.addLayer(image, {'bands': ['B4', 'B3', 'B2'], 'max': 0.4}, '原始影像')
                Map.addLayer(classified,
                           {'min': 0, 'max': 2, 'palette': ['red', 'green', 'blue']},
                           '分类结果')
                
                # 缩放到训练样本区域
                Map.centerObject(points, 11)
                
                st.success("随机森林分类完成！")
                
            except Exception as e:
                st.error(f"分类失败: {str(e)}")

# 左侧地图显示
with col1:
    Map.to_streamlit(height=850)

# 页脚信息
st.markdown("---")
st.markdown("© 2024 GEE影像随机森林分类器 | 使用Google Earth Engine和Geemap构建") 