"""
============================================================
阶段 3C+：Folium 交互式县域专题图
============================================================
用已验证的周口多指数数据 + Folium
产出可缩放/点击/切换指数的交互式 HTML 地图
============================================================
"""
import os, json, warnings
warnings.filterwarnings("ignore")

os.environ["CURL_CA_BUNDLE"] = ""

import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium import Choropleth, LayerControl, GeoJsonTooltip
from folium.plugins import Search

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "outputs", "zhoukou_multi")

# ===== 加载已验证数据 =====
zk_tif = os.path.join(OUT, "zk_indices.shp")
geojson_path = os.path.join(OUT, "counties_zk.geojson")

# 读取县域
gdf = gpd.read_file(zk_tif).to_crs("EPSG:4326")
print(f"✅ 县域: {len(gdf)} 个")
print(gdf.columns.tolist())

# ===== 基础地图 =====
center_lat, center_lon = 33.57, 114.75   # 周口中心

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=11,
    tiles="OpenStreetMap",
    control_scale=True,
    prefer_canvas=True,
)

# ===== 指数配置 =====
INDEX_CONFIG = {
    "ndvi": {
        "label": "NDVI 植被绿度",
        "bins": [0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "colors": ["#d73027", "#fdae61", "#fee08b", "#d9ef8b", "#1a9850"],
        "vmin": -0.2, "vmax": 0.8,
    },
    "evi": {
        "label": "EVI 增强植被指数",
        "bins": [0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "colors": ["#d73027", "#fdae61", "#fee08b", "#d9ef8b", "#1a9850"],
        "vmin": -0.2, "vmax": 0.8,
    },
    "savi": {
        "label": "SAVI 土壤调节指数",
        "bins": [0, 0.15, 0.3, 0.45, 0.6],
        "colors": ["#d73027", "#fdae61", "#fee08b", "#d9ef8b", "#1a9850"],
        "vmin": -0.2, "vmax": 0.6,
    },
    "ndwi": {
        "label": "NDWI 水体指数",
        "bins": [-0.5, -0.25, 0, 0.25, 0.5],
        "colors": ["#fee0d2", "#fcbba1", "#6baed6", "#2171b5", "#08519c"],
        "vmin": -0.5, "vmax": 0.5,
    },
}

# ===== 为每个指数加 Choropleth 图层 =====
for idx, cfg in INDEX_CONFIG.items():
    col_mean = f"{idx}_mean"

    Choropleth(
        geo_data=gdf,
        name=cfg["label"],
        data=gdf,
        columns=["name", col_mean],
        key_on="feature.properties.name",
        fill_color="YlGn" if idx != "ndwi" else "Blues",
        fill_opacity=0.7,
        line_opacity=0.5,
        legend_name=cfg["label"],
        vmin=cfg["vmin"],
        vmax=cfg["vmax"],
        control=True,
        show=(idx == "ndvi"),   # 默认只显示 NDVI
    ).add_to(m)

# ===== GeoJSON + Tooltip =====
gdf_json = json.loads(gdf.to_json())

geojson_layer = folium.GeoJson(
    gdf_json,
    name="县域边界",
    style_function=lambda x: {"fillOpacity": 0, "weight": 1, "color": "#555"},
    tooltip=GeoJsonTooltip(
        fields=["name", "ndvi_mean", "evi_mean", "savi_mean", "ndwi_mean",
                "ndvi_std", "ndvi_count"],
        aliases=["行政区:", "NDVI 均值:", "EVI 均值:", "SAVI 均值:", "NDWI 均值:",
                "NDVI 标准差:", "有效像元数:"],
        localize=True,
        sticky=True,
    ),
    control=True,
    show=True,
)
geojson_layer.add_to(m)

# ===== Layer Control =====
LayerControl(collapsed=False, position="topright").add_to(m)

# ===== 标题 =====
from folium.plugins import Fullscreen
Fullscreen().add_to(m)

# ===== 保存 =====
html_path = os.path.join(OUT, "zhoukou_interactive_map.html")
m.save(html_path)
print(f"\n✅ 交互式地图已保存: {html_path}")
print(f"   在浏览器打开即可：")
print(f"   file:///{html_path.replace(os.sep, '/')}")
print(f"\n功能：")
print(f"   🗺️  缩放/拖拽浏览")
print(f"   👆  悬停显示县域统计值")
print(f"   🎚️  右上角切换 4 个指数图层")
print(f"   📱  全屏按钮")
