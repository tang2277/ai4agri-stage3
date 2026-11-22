import os, warnings
warnings.filterwarnings("ignore")
os.environ["PROJ_LIB"] = r"C:\Users\Leng\miniconda3\envs\ai4agri\lib\site-packages\pyproj\proj_dir\share\proj"
os.environ["PROJ_DATA"] = os.environ["PROJ_LIB"]

from pyproj import CRS
import geopandas as gpd
import rasterio
import numpy as np

# 河南县域
shp = r"C:\Users\Leng\Documents\Trae-CN\default\projects\remote-sensing-gis\outputs\henan_real\henan_counties.shp"
gdf = gpd.read_file(shp)
print("="*60)
print("📦 河南县域 Shapefile 详情")
print("="*60)
print(f"要素数: {len(gdf)}")
print(f"CRS: {gdf.crs}")
print(f"字段: {list(gdf.columns)}")
print(f"\n前 5 个行政区:")
for _, r in gdf.head(5).iterrows():
    print(f"  {r.get('name','?')} ({r.get('adcode','?')}) level={r.get('level','?')}")

# 统计 level 分布
if 'level' in gdf.columns:
    print(f"\nlevel 分布:")
    print(gdf['level'].value_counts())

# Sentinel-2 NDVI
print("\n" + "="*60)
print("🛰️  Sentinel-2 NDVI 详情 (周口区域)")
print("="*60)
ndvi_tif = r"C:\Users\Leng\Documents\Trae-CN\default\projects\remote-sensing-gis\outputs\sentinel2_real\real_s2_ndvi.tif"
with rasterio.open(ndvi_tif) as src:
    ndvi = src.read(1)
    print(f"shape: {ndvi.shape}")
    print(f"CRS: {src.crs}")
    print(f"分辨率: {src.res}")
    valid = ndvi[~np.isnan(ndvi)]
    print(f"均值 NDVI: {np.nanmean(valid):.4f}")
    print(f"中位数 NDVI: {np.nanmedian(valid):.4f}")
    print(f"范围: [{np.nanmin(valid):.4f}, {np.nanmax(valid):.4f}]")
    print(f"标准差: {np.nanstd(valid):.4f}")
    
    # 阈值分布
    print(f"\nNDVI 分类分布:")
    print(f"  < 0    (水体/裸土): {np.mean(valid < 0)*100:.1f}%")
    print(f"  0-0.2  (稀疏植被): {np.mean((valid >= 0) & (valid < 0.2))*100:.1f}%")
    print(f"  0.2-0.4 (中等植被): {np.mean((valid >= 0.2) & (valid < 0.4))*100:.1f}%")
    print(f"  0.4-0.6 (茂盛植被): {np.mean((valid >= 0.4) & (valid < 0.6))*100:.1f}%")
    print(f"  > 0.6  (密林/作物): {np.mean(valid >= 0.6)*100:.1f}%")
