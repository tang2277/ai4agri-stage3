"""
===========================================================
阶段 3B · 遥感 + GIS 完整管线
===========================================================
功能：
  1. 模拟 Sentinel-2 多光谱影像（带 WGS84 地理坐标）
  2. 计算 NDVI 并保存为 GeoTIFF
  3. 生成县域边界 shapefile（模拟河南周口 4 县）
  4. 县域级 NDVI 分区统计（zonal statistics）
  5. GeoPandas 专题图可视化
  6. Folium 交互式地图（可在浏览器打开）

技术栈：rasterio + geopandas + folium + matplotlib + numpy
时间：约 30 秒（纯 CPU）
===========================================================
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# 修复 Windows pyproj CRS 数据库路径
os.environ.setdefault("PROJ_LIB", os.path.join(os.path.dirname(sys.executable), "Library", "share", "proj"))
os.environ.setdefault("GDAL_DATA", os.path.join(os.path.dirname(sys.executable), "Library", "share", "gdal"))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import rasterio
from rasterio.transform import from_bounds
from rasterio.mask import mask as rio_mask
import geopandas as gpd
from shapely.geometry import Polygon, box
import folium
from rasterstats import zonal_stats

# ========== 配置 ==========
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "outputs")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

CRS = 4326  # WGS84（用数字绕过 pyproj CRS 解析问题）
print(f"📁 项目目录: {BASE_DIR}")
print(f"📁 输出目录: {OUT_DIR}")

# ============================================================
# Step 1: 生成模拟 Sentinel-2 多光谱影像
# ============================================================
print("\n" + "="*60)
print("🔄 Step 1: 生成模拟 Sentinel-2 多光谱影像")
print("="*60)

H, W = 500, 500
np.random.seed(42)

# 模拟河南周口附近 4 个县域地块
# 地理位置：东经 114.65°-114.85°，北纬 33.50°-33.65°
lon_min, lon_max = 114.65, 114.85
lat_min, lat_max = 33.50, 33.65

transform = from_bounds(lon_min, lat_min, lon_max, lat_max, W, H)

# 用 meshgrid 生成地块
yy, xx = np.mgrid[0:H, 0:W]

# 划分 4 个县域（模拟）
county = np.zeros((H, W), dtype=np.uint8)
mid_h, mid_w = H // 2, W // 2
county[:mid_h, :mid_w] = 1   # 西华县（小麦）
county[:mid_h, mid_w:] = 2   # 商水县（玉米）
county[mid_h:, :mid_w] = 3   # 淮阳县（大豆+病害）
county[mid_h:, mid_w:] = 4   # 太康县（裸土/休耕）

# 各 county 的 NDVI 特征
county_profiles = {
    0: (0.10, 0.15),   # 背景
    1: (0.08, 0.72),   # 西华县 小麦 长势好  NDVI≈0.79
    2: (0.15, 0.50),   # 商水县 玉米 长势中  NDVI≈0.54
    3: (0.28, 0.32),   # 淮阳县 大豆 病害多  NDVI≈0.07
    4: (0.32, 0.38),   # 太康县 休耕 裸土   NDVI≈0.09
}

# 生成 red / nir band
red_band = np.zeros((H, W))
nir_band = np.zeros((H, W))
noise_level = 0.04

for cid in range(5):
    mask = county == cid
    if mask.sum() > 0:
        red_val, nir_val = county_profiles[cid]
        red_band[mask] = np.random.normal(red_val, noise_level, mask.sum())
        nir_band[mask] = np.random.normal(nir_val, noise_level, mask.sum())

red_band = np.clip(red_band, 0, 1).astype(np.float32)
nir_band = np.clip(nir_band, 0, 1).astype(np.float32)

print(f"  ✅ Red band: shape={red_band.shape}, range=[{red_band.min():.3f}, {red_band.max():.3f}]")
print(f"  ✅ NIR band: shape={nir_band.shape}, range=[{nir_band.min():.3f}, {nir_band.max():.3f}]")

# ============================================================
# Step 2: 保存 GeoTIFF + 计算 NDVI
# ============================================================
print("\n" + "="*60)
print("💾 Step 2: 保存 GeoTIFF + 计算 NDVI")
print("="*60)

for band_data, band_name in [(red_band, "B04_red"), (nir_band, "B08_nir")]:
    tif_path = os.path.join(OUT_DIR, f"sentinel2_{band_name}.tif")
    with rasterio.open(
        tif_path, "w", driver="GTiff", height=H, width=W,
        count=1, dtype="float32", crs=CRS, transform=transform,
    ) as dst:
        dst.write(band_data, 1)
    print(f"  ✅ {os.path.basename(tif_path)}  CRS={CRS}")

# 计算 NDVI
ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-8)
ndvi = np.clip(ndvi, -1, 1).astype(np.float32)

# 保存 NDVI GeoTIFF
ndvi_tif = os.path.join(OUT_DIR, "ndvi.tif")
with rasterio.open(
    ndvi_tif, "w", driver="GTiff", height=H, width=W,
    count=1, dtype="float32", crs=CRS, transform=transform,
) as dst:
    dst.write(ndvi, 1)
print(f"  ✅ ndvi.tif  NDVI 均值={ndvi.mean():.3f}, 范围=[{ndvi.min():.3f}, {ndvi.max():.3f}]")

# ============================================================
# Step 3: 生成县域边界 Shapefile
# ============================================================
print("\n" + "="*60)
print("🗺️ Step 3: 生成县域边界 Shapefile")
print("="*60)

# 把像素坐标转地理坐标，生成县域 polygon
def pixel_to_geo(px, py):
    x, y = rasterio.transform.xy(transform, py, px)
    return x, y

county_names = {
    1: "西华县 Xihua",
    2: "商水县 Shangshui",
    3: "淮阳县 Huaiyang",
    4: "太康县 Taikang",
}

polygons = []
for cid in range(1, 5):
    # 找到该 county 的边界像素
    mask = county == cid
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # 转地理坐标
    x1, y1 = pixel_to_geo(cmin, rmin)
    x2, y2 = pixel_to_geo(cmax, rmax)
    
    poly = box(x1, y1, x2, y2)
    polygons.append(poly)

# 加一点边界噪动让形状更真实
import random
random.seed(42)
for i, p in enumerate(polygons):
    coords = list(p.exterior.coords)
    # 把 rectangles 转成有起伏的 polygon
    jittered = [(c[0] + random.uniform(-0.003, 0.003), 
                 c[1] + random.uniform(-0.003, 0.003)) for c in coords]
    polygons[i] = Polygon(jittered)

gdf = gpd.GeoDataFrame({
    "county_id": range(1, 5),
    "county": [county_names[i] for i in range(1, 5)],
    "geometry": polygons,
})

shp_path = os.path.join(DATA_DIR, "zhoukou_counties.shp")
gdf.to_file(shp_path, driver="ESRI Shapefile")

# 手动写 .prj 文件（WGS84 / EPSG:4326）绕过 pyproj CRS 问题
prj_path = shp_path.replace(".shp", ".prj")
wgs84_wkt = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'
with open(prj_path, "w") as f:
    f.write(wgs84_wkt)
print(f"  ✅ 县域 Shapefile: {shp_path}")
print(f"     {len(gdf)} 个县域: {list(gdf['county'])}")

# ============================================================
# Step 4: Zonal Statistics 县域级 NDVI 统计
# ============================================================
print("\n" + "="*60)
print("📊 Step 4: Zonal Statistics 县域级 NDVI 统计")
print("="*60)

stats = zonal_stats(shp_path, ndvi_tif, 
                    stats=["min", "max", "mean", "std", "median"])

gdf["ndvi_mean"] = [s["mean"] for s in stats]
gdf["ndvi_std"] = [s["std"] for s in stats]
gdf["ndvi_median"] = [s["median"] for s in stats]

# 长势评价
def evaluate(ndvi):
    if ndvi > 0.5: return "✅ 长势良好", "#2E7D32"
    elif ndvi > 0.3: return "⚠️ 长势一般", "#F9A825"
    elif ndvi > 0.1: return "❌ 长势较差", "#E64A19"
    else: return "🔴 病害/裸土", "#B71C1C"

gdf["status"], gdf["color"] = zip(*[evaluate(m) for m in gdf["ndvi_mean"]])

print(f"\n  {'县域':<20} {'平均NDVI':<10} {'中位数':<10} {'长势评价'}")
print("  " + "-"*65)
for _, row in gdf.iterrows():
    print(f"  {row['county']:<20} {row['ndvi_mean']:<10.3f} {row['ndvi_median']:<10.3f} {row['status']}")

# ============================================================
# Step 5: GeoPandas 专题图
# ============================================================
print("\n" + "="*60)
print("🎨 Step 5: 生成 GeoPandas 专题图")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle("河南省周口市 · 县域 NDVI 作物长势专题图\n(模拟数据 · 2024 小麦季)", 
             fontsize=16, fontweight="bold")

# 左图：NDVI 栅格 + 县域边界
ndvi_cmap = LinearSegmentedColormap.from_list("ndvi", [
    "#B71C1C", "#E64A19", "#F9A825", "#7CB518", "#2E7D32"
])

with rasterio.open(ndvi_tif) as src:
    axes[0].imshow(src.read(1), cmap=ndvi_cmap, vmin=-0.2, vmax=0.8, extent=[lon_min, lon_max, lat_min, lat_max])

gdf.boundary.plot(ax=axes[0], color="black", linewidth=1.2)
gdf.apply(lambda r: axes[0].text(r.geometry.centroid.x, r.geometry.centroid.y, 
                                  r["county"].split(" ")[0], 
                                  ha="center", va="center", fontsize=10, fontweight="bold", color="white"), axis=1)
axes[0].set_title("NDVI 遥感影像 + 县域边界")
axes[0].set_xlabel("经度 (°E)")
axes[0].set_ylabel("纬度 (°N)")

# 右图：县域专题图（按 NDVI 平均着色）
gdf.plot(ax=axes[1], column="ndvi_mean", cmap=ndvi_cmap, vmin=-0.2, vmax=0.8, edgecolor="black", linewidth=1.2)
gdf.apply(lambda r: axes[1].text(r.geometry.centroid.x, r.geometry.centroid.y, 
                                 f"{r['county'].split(' ')[0]}\n{r['ndvi_mean']:.2f}", 
                                 ha="center", va="center", fontsize=11, fontweight="bold"), axis=1)
axes[1].set_title("县域平均 NDVI 专题图")
axes[1].set_xlabel("经度 (°E)")
axes[1].set_ylabel("纬度 (°N)")
axes[1].set_xticks([])
axes[1].set_yticks([])

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "county_ndvi_map.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✅ 专题图已保存: {out_path}")

# ============================================================
# Step 6: Folium 交互式地图
# ============================================================
print("\n" + "="*60)
print("🗺️ Step 6: 生成 Folium 交互式地图")
print("="*60)

m = folium.Map(location=[(lat_min + lat_max)/2, (lon_min + lon_max)/2], 
               zoom_start=11, tiles="OpenStreetMap")

# 添加 NDVI 栅格 overlay
with rasterio.open(ndvi_tif) as src:
    bounds = rasterio.transform.array_bounds(H, W, src.transform)
    bounds_ll = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]

folium.raster_layers.ImageOverlay(
    image=ndvi,
    bounds=bounds_ll,
    colormap=lambda x: ndvi_cmap(np.clip(x, -0.2, 0.8)),
    opacity=0.7,
    name="NDVI 作物长势",
).add_to(m)

# 添加县域边界 + tooltip
for _, row in gdf.iterrows():
    geojson = folium.GeoJson(
        row["geometry"],
        style_function=lambda f, c=row["color"]: {
            "fillColor": c, "color": "black", "weight": 2, "fillOpacity": 0.6
        }
    )
    tooltip = folium.Tooltip(
        f"<b>{row['county']}</b><br>"
        f"平均 NDVI: {row['ndvi_mean']:.3f}<br>"
        f"中位数: {row['ndvi_median']:.3f}<br>"
        f"标准差: {row['ndvi_std']:.3f}<br>"
        f"长势: {row['status']}"
    )
    geojson.add_child(tooltip)
    geojson.add_to(m)

# Layer control
folium.LayerControl().add_to(m)

html_path = os.path.join(OUT_DIR, "interactive_map.html")
m.save(html_path)
print(f"  ✅ 交互式地图已保存: {html_path}")
print(f"  💡 直接用浏览器打开即可交互查看！")

# ============================================================
# 完成总结
# ============================================================
print("\n" + "="*60)
print("🎉 阶段 3B 遥感 + GIS 完整管线 跑完啦！")
print("="*60)
print(f"\n📦 产出文件:")
for f in sorted(os.listdir(OUT_DIR)):
    size = os.path.getsize(os.path.join(OUT_DIR, f))
    if size > 1024*1024:
        print(f"  📄 {OUT_DIR}/{f}  ({size/1024/1024:.1f} MB)")
    else:
        print(f"  📄 {OUT_DIR}/{f}  ({size/1024:.1f} KB)")

print(f"\n🎯 核心产出:")
print(f"  ① ndvi.tif              — NDVI GeoTIFF（可在 QGIS 打开）")
print(f"  ② county_ndvi_map.png   — 县域专题图（左：影像 右：县域着色）")
print(f"  ③ interactive_map.html   — Folium 交互式地图（浏览器打开）")
print(f"  ④ data/zhoukou_counties.shp — 县域边界 Shapefile")

print(f"\n🚀 下一步:")
print(f"  1. 双击 interactive_map.html → 浏览器交互查看")
print(f"  2. 把 GeoTIFF / Shapefile 拖进 QGIS → 桌面端可视化")
print(f"  3. Google Earth Engine 注册 → 下载真实 Sentinel-2 数据替换模拟")
