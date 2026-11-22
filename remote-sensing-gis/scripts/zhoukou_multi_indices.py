"""
============================================================
阶段 3C：周口区域 · 多指数管线（最小验证版）
============================================================
复用已验证的周口 bbox [114.65, 33.50, 114.85, 33.65]
之前 download_real_ndvi.py 在这个范围跑通了
验证：多指数 + Zonal Stats + 专题图 全流程
============================================================
"""
import os, sys, json, warnings, gc
warnings.filterwarnings("ignore")
os.environ["CURL_CA_BUNDLE"] = ""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from shapely.geometry import shape, box
import requests

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "outputs", "zhoukou_multi")
os.makedirs(OUT, exist_ok=True)

# ===== 已验证的周口 bbox =====
ZK_BBOX = [114.65, 33.50, 114.85, 33.65]

print("=" * 65)
print("🛰️  阶段 3C 周口验证版：NDVI / EVI / SAVI / NDWI")
print("=" * 65)

# ==============================
# Step 1: 县域边界 + 过滤
# ==============================
print("\n📥 Step 1: 县域边界...")
resp = requests.get("https://geo.datav.aliyun.com/areas_v3/bound/410000_full.json", timeout=30)
gj = resp.json()

all_features = []
def flatten(features):
    for f in features:
        all_features.append(f)
        children = f.get("properties", {}).get("children", [])
        if children:
            flatten(children)
flatten(gj["features"])

counties = [f for f in all_features if str(f["properties"]["adcode"]).startswith("41")]
geoms = [shape(f["geometry"]) for f in counties]
props = [{k: v for k, v in f["properties"].items()
          if k not in ("center", "centroid", "children", "acroutes", "childrenNum")}
         for f in counties]

gdf = gpd.GeoDataFrame(props, geometry=geoms, crs="EPSG:4326")

# 过滤到周口 bbox
zk_poly = box(*ZK_BBOX)
gdf_zk = gdf[gdf.geometry.intersects(zk_poly)].copy()
print(f"  ✅ 河南 {len(gdf)} 个行政区 → 周口范围内 {len(gdf_zk)} 个")

gdf_zk.to_file(os.path.join(OUT, "counties_zk.geojson"), driver="GeoJSON")

# ==============================
# Step 2: 搜索 + 裁剪
# ==============================
print("\n🛰️  Step 2: STAC 搜索 + 裁剪...")
from pystac_client import Client
import planetary_computer as pc

api = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
search = api.search(
    collections=["sentinel-2-l2a"],
    bbox=ZK_BBOX,
    datetime="2024-04-01/2024-05-31",
    query={"eo:cloud_cover": {"lt": 30}},
    sortby="-properties.eo:cloud_cover",
    limit=10,
)
items = [pc.sign(it) for it in search.items()]
print(f"  ✅ 找到 {len(items)} 景")

# 选云量最低的 1 景
it = items[0]
tile = it.properties.get("s2:mgrs_tile", "?")
cc = it.properties.get("eo:cloud_cover", 0)
dt = it.datetime.date()
print(f"  🎯 选中: {tile} | {dt} | 云量 {cc:.1f}%")

# ===== 先开 B04 算裁剪窗口 =====
print(f"\n  🔪 裁剪 4 个波段...")
BANDS = {"B02": "Blue", "B03": "Green", "B04": "Red", "B08": "NIR"}

with rasterio.open("/vsicurl/" + it.assets["B04"].href, timeout=120) as src:
    print(f"     B04 打开: shape={src.shape}, crs={src.crs}")
    utm_bbox = transform_bounds("EPSG:4326", src.crs, *ZK_BBOX)
    w = from_bounds(*utm_bbox, src.transform)
    print(f"     裁剪窗口: {int(w.width)}×{int(w.height)} 像元")

# ===== 逐波段裁剪 =====
arrays = {}
for bk, bn in BANDS.items():
    t0 = __import__('time').time()
    url = "/vsicurl/" + it.assets[bk].href
    with rasterio.open(url, timeout=120) as src:
        arr = src.read(1, window=w).astype('float32') / 10000.0
        arrays[bk] = arr
    print(f"     ✅ {bk} ({bn}): {arr.shape} in {__import__('time').time()-t0:.1f}s")

# ==============================
# Step 3: 多指数计算
# ==============================
print("\n🧮 Step 3: 多指数计算...")
red, nir, blue, green = arrays["B04"], arrays["B08"], arrays["B02"], arrays["B03"]

ndvi = np.clip((nir - red) / (nir + red + 1e-8), -1, 1)
evi = np.clip(2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1 + 1e-8), -1, 1)
savi = np.clip((nir - red) / (nir + red + 0.5) * 1.5, -1, 1)
ndwi = np.clip((green - nir) / (green + nir + 1e-8), -1, 1)
del arrays, red, nir, blue, green; gc.collect()

for n, a in [("NDVI", ndvi), ("EVI", evi), ("SAVI", savi), ("NDWI", ndwi)]:
    print(f"  🧮 {n}: 均值={np.nanmean(a):.4f} | [{np.nanmin(a):.3f}, {np.nanmax(a):.3f}]")

# 写 tif
profile = {"driver": "GTiff", "height": ndvi.shape[0], "width": ndvi.shape[1],
           "count": 1, "dtype": "float32", "crs": src.crs,
           "transform": src.window_transform(w)}

for n, a in [("ndvi", ndvi), ("evi", evi), ("savi", savi), ("ndwi", ndwi)]:
    with rasterio.open(os.path.join(OUT, f"zk_{n}.tif"), "w", **profile) as dst:
        dst.write(a, 1)
print(f"  💾 4 个 tif 已保存")

# ==============================
# Step 4: Zonal Statistics
# ==============================
print("\n📊 Step 4: Zonal Statistics...")
from rasterstats import zonal_stats

gdf_proj = gdf_zk.to_crs(src.crs)

indices = ["ndvi", "evi", "savi", "ndwi"]
stats_all = {}

for idx in indices:
    stats = zonal_stats(gdf_proj, os.path.join(OUT, f"zk_{idx}.tif"),
                       stats=["mean", "std", "min", "max", "count"],
                       all_touched=True, nodata=np.nan)
    stats_all[idx] = stats
    valid = sum(1 for s in stats if s["mean"] is not None and s["count"] > 0)
    print(f"  ✅ {idx.upper()}: {valid}/{len(stats)} 有数据")

# 合并
gdf_out = gdf_proj.reset_index(drop=True)
for idx in indices:
    for col in ["mean", "std", "min", "max", "count"]:
        vals = [s[col] if s else np.nan for s in stats_all[idx]]
        gdf_out[f"{idx}_{col}"] = vals

valid = gdf_out[gdf_out["ndvi_mean"].notna() & (gdf_out["ndvi_count"] > 0)]
print(f"  📋 {len(valid)} 个行政区有完整数据")

print("\n  🥇 按 NDVI 排序:")
for _, r in valid.sort_values("ndvi_mean", ascending=False).iterrows():
    name = r.get('name', '?')
    print(f"     {name:12s}  NDVI={r['ndvi_mean']:.3f}  EVI={r['evi_mean']:.3f}  SAVI={r['savi_mean']:.3f}  NDWI={r['ndwi_mean']:.3f}")

valid.to_file(os.path.join(OUT, "zk_indices.shp"), driver="ESRI Shapefile")

# ==============================
# Step 5: 专题图
# ==============================
print("\n🎨 Step 5: 专题图...")
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

configs = [
    (ndvi, "ndvi_mean", "NDVI", "RdYlGn", -0.2, 0.8),
    (evi,  "evi_mean",  "EVI",  "RdYlGn", -0.2, 0.8),
    (savi, "savi_mean", "SAVI", "RdYlGn", -0.2, 0.8),
    (ndwi, "ndwi_mean", "NDWI", "Blues",  -0.5, 0.5),
]

for idx, (arr, col, title, cmap, vmin, vmax) in enumerate(configs):
    r, c = idx // 2, idx % 2
    ax = axes[r, c]

    # 栅格背景
    step = max(1, arr.shape[0] // 500)
    small = arr[::step, ::step]
    im = ax.imshow(small, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(f"{title} · 周口区域", fontsize=12, fontweight="bold")
    ax.axis("off")

    # 县域叠层
    if len(valid) > 0:
        valid.plot(column=col, ax=ax, cmap=cmap,
                    edgecolor="white", linewidth=0.5,
                    vmin=vmin, vmax=vmax, alpha=0.8)

    plt.colorbar(im, ax=ax, shrink=0.7, label=title)

plt.suptitle("周口区域 · 2024 小麦季 Sentinel-2 多指数专题图",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "zk_multi_map.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"  💾 zk_multi_map.png")

# ==============================
# 指数对比表
# ==============================
print("\n📊 县域指数均值对比:")
print("-" * 50)
print(f"{'指数':<8} {'均值':>10} {'标准差':>10} {'最小':>10} {'最大':>10}")
print("-" * 50)
for idx in indices:
    m = np.nanmean([r[f"{idx}_mean"] for _, r in valid.iterrows()])
    s = np.nanmean([r[f"{idx}_std"] for _, r in valid.iterrows()])
    mn = np.nanmin([r[f"{idx}_min"] for _, r in valid.iterrows()])
    mx = np.nanmax([r[f"{idx}_max"] for _, r in valid.iterrows()])
    print(f"{idx.upper():<8} {m:>10.4f} {s:>10.4f} {mn:>10.4f} {mx:>10.4f}")

print("\n" + "=" * 65)
print("🎉 阶段 3C 周口验证版完成！")
print(f"\n📦 产出 ({OUT}/):")
print(f"  zk_ndvi.tif / evi.tif / savi.tif / ndwi.tif")
print(f"  zk_indices.shp         县域 + 4 指数统计")
print(f"  counties_zk.geojson    周口范围边界")
print(f"  zk_multi_map.png       🎨 2×2 专题图")
