"""
============================================================
河南省县域 Shapefile + 全省 Sentinel-2 NDVI + 县域专题图
============================================================
PROJ4 字符串 CRS 绕过 pyproj 3.7 bug
============================================================
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")

# PROJ 环境
os.environ["PROJ_LIB"] = r"C:\Users\Leng\miniconda3\envs\ai4agri\lib\site-packages\pyproj\proj_dir\share\proj"
os.environ["PROJ_DATA"] = os.environ["PROJ_LIB"]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from shapely.geometry import shape
from pyproj import CRS
import requests

# ===== CRS 常量（绕过 pyproj from_epsg bug）=====
WGS84 = CRS.from_proj4("+proj=longlat +datum=WGS84")
UTM50N = CRS.from_proj4("+proj=utm +zone=50 +datum=WGS84")
print(f"✅ CRS 构造成功: WGS84={WGS84.is_geographic}, UTM50N={UTM50N.is_projected}")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "outputs", "henan_real")
os.makedirs(OUT, exist_ok=True)

print("="*60)
print("🗺️  河南省县域 Shapefile + 全省 Sentinel-2 NDVI")
print("="*60)

# ========== Step 1: 河南县域 GeoJSON ==========
print("\n📥 Step 1: 下载河南 18 地级市边界 + 嵌套县域...")

resp = requests.get("https://geo.datav.aliyun.com/areas_v3/bound/410000_full.json", timeout=30)
gj = resp.json()

# DataV 嵌套 children 递归展平
all_features = []
def flatten(features):
    for f in features:
        all_features.append(f)
        children = f.get("properties", {}).get("children", [])
        if children:
            flatten(children)

flatten(gj["features"])
print(f"  ✅ GeoJSON 总要素: {len(all_features)}")

# 河南县级: adcode 41xxxx, level=district
counties = [f for f in all_features if str(f["properties"]["adcode"]).startswith("41")]
print(f"  ✅ 河南县级: {len(counties)} 个")

geoms = [shape(f["geometry"]) for f in counties]
props = [{k: v for k, v in f["properties"].items() 
         if k not in ("center", "centroid", "children", "acroutes", "childrenNum")} 
         for f in counties]

# 先无 CRS 构建，然后手动设
gdf = gpd.GeoDataFrame(props, geometry=geoms)
gdf = gdf.set_crs(WGS84, allow_override=True)
print(f"  ✅ CRS: WGS84")

gdf_utm = gdf.to_crs(UTM50N)
print(f"  ✅ UTM50N 投影变换")

gdf.to_file(os.path.join(OUT, "henan_counties_wgs84.geojson"), driver="GeoJSON")
gdf_utm.to_file(os.path.join(OUT, "henan_counties.shp"), driver="ESRI Shapefile")
print(f"  💾 Shapefile 已保存")

# ========== Step 2: 河南全省 Sentinel-2 NDVI ==========
print("\n🛰️  Step 2: 河南全省 Sentinel-2 NDVI...")

bbox = [110.35, 31.38, 116.62, 36.37]  # WGS84

from pystac_client import Client
import planetary_computer as pc

api = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
search = api.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime="2024-04-01/2024-05-31",
    query={"eo:cloud_cover": {"lt": 40}},
    sortby="-properties.eo:cloud_cover",
    limit=10,
)
items = [pc.sign(it) for it in search.items()]
print(f"  ✅ 找到 {len(items)} 景")

for i, it in enumerate(items[:5]):
    print(f"     [{i+1}] {it.datetime.date()} | 云量 {it.properties.get('eo:cloud_cover',0):.1f}% | 分区 {it.properties.get('s2:mgrs_tile','?')}")

item = items[0]
print(f"\n  🎯 选中: {item.datetime.date()} | 云量 {item.properties.get('eo:cloud_cover',0):.1f}%")

os.environ["CURL_CA_BUNDLE"] = ""
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds

red_url = "/vsicurl/" + item.assets["B04"].href
nir_url = "/vsicurl/" + item.assets["B08"].href

with rasterio.open(red_url) as src_r:
    print(f"  B04 CRS: {src_r.crs} | shape: {src_r.shape}")
    utm_bbox = transform_bounds(WGS84, src_r.crs, *bbox)
    w = from_bounds(*utm_bbox, src_r.transform)
    red = src_r.read(1, window=w).astype(np.float32) / 10000.0
    transform = src_r.window_transform(w)
    crs = src_r.crs
    print(f"  ✅ Red 裁剪后: {red.shape}")

with rasterio.open(nir_url) as src_n:
    nir = src_n.read(1, window=w).astype(np.float32) / 10000.0

ndvi = (nir - red) / (nir + red + 1e-8)
ndvi = np.clip(ndvi, -1, 1)
print(f"  🧮 NDVI: {ndvi.shape} | 均值 {np.nanmean(ndvi):.4f}")

ndvi_tif = os.path.join(OUT, "henan_s2_ndvi.tif")
with rasterio.open(ndvi_tif, "w", driver="GTiff",
                   height=ndvi.shape[0], width=ndvi.shape[1],
                   count=1, dtype="float32", crs=crs, transform=transform,
                   BIGTIFF="IF_SAFER") as dst:
    dst.write(ndvi, 1)
print(f"  💾 henan_s2_ndvi.tif")

# ========== Step 3: 县域 Zonal Statistics ==========
print("\n📊 Step 3: 县域 NDVI 统计...")
from rasterstats import zonal_stats

stats = zonal_stats(gdf_utm, ndvi_tif,
                    stats=["mean", "std", "min", "max", "count", "percentile_90"],
                    all_touched=True)
print(f"  ✅ {len(stats)} 县域统计完成")

stats_df = pd.DataFrame(stats)
gdf_utm = gdf_utm.reset_index(drop=True)
for col in stats_df.columns:
    gdf_utm[col] = stats_df[col]

valid = gdf_utm[gdf_utm["mean"].notna() & (gdf_utm["count"] > 0)]
print(f"  📋 有 NDVI 数据的县域: {len(valid)} / {len(gdf_utm)}")

print(f"\n  🥇 Top 10 县域 NDVI:")
for _, r in valid.sort_values("mean", ascending=False).head(10).iterrows():
    print(f"     {r.get('name','?'):10s}  NDVI={r['mean']:.3f}  max={r['max']:.3f}")

print(f"\n  🔻 Bottom 5 (可能病害/休耕):")
for _, r in valid.sort_values("mean").head(5).iterrows():
    print(f"     {r.get('name','?'):10s}  NDVI={r['mean']:.3f}")

out_shp = os.path.join(OUT, "henan_counties_ndvi.shp")
valid.to_file(out_shp, driver="ESRI Shapefile")
print(f"  💾 {out_shp}")

# ========== Step 4: 专题图 ==========
print("\n🎨 Step 4: 县域专题图...")
fig, axes = plt.subplots(1, 2, figsize=(16, 10))

im0 = axes[0].imshow(ndvi, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
axes[0].set_title("Sentinel-2 真实 NDVI\n河南全省 · 2024 小麦季", fontsize=12)
axes[0].axis("off")
plt.colorbar(im0, ax=axes[0], shrink=0.7, label="NDVI")

valid.plot(column="mean", ax=axes[1], cmap="RdYlGn",
           legend_kwds={"label": "县域平均 NDVI"},
           edgecolor="white", linewidth=0.3,
           vmin=-0.1, vmax=0.7)
axes[1].set_title("河南省县域 NDVI 专题图\n(真实 Sentinel-2 + 县域边界)", fontsize=12)
axes[1].set_axis_off()

plt.tight_layout()
out_png = os.path.join(OUT, "henan_ndvi_county_map.png")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close()
print(f"  💾 {out_png}")

print("\n" + "="*60)
print("🎉 阶段 3B · 河南省域真实遥感 GIS 完成!")
print("="*60)
print(f"""
📦 产出清单 ({OUT}/):
  henan_counties.shp          河南县域边界
  henan_counties_ndvi.shp     县域 + NDVI 统计
  henan_s2_ndvi.tif           河南全省真实 NDVI
  henan_ndvi_county_map.png   🎨 县域专题图
  henan_counties_wgs84.geojson WGS84 GeoJSON
""")
