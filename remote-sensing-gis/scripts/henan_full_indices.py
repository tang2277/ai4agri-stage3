"""
============================================================
阶段 3C：河南省域 · 多指数遥感 GIS 管线
============================================================
✅ 已修复 PROJ 数据库冲突（不设 PROJ_LIB，用 rasterio 自带）
✅ 已验证远程 COG 裁剪（小窗口 < 5km 可接受）
指数：NDVI / EVI / SAVI / NDWI
============================================================
"""
import os, sys, json, warnings, gc
warnings.filterwarnings("ignore")

# ===== 关键修复：不要设 PROJ_LIB！=====
# pyproj 的 proj.db 是 layout v4，rasterio 1.4.3 / GDAL 3.10.3 要 v5+
# 设了会让 rasterio 的 transform_bounds 崩
os.environ["CURL_CA_BUNDLE"] = ""   # Windows SSL fix

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds, Window
from rasterio.warp import transform_bounds
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import shape, box
import requests

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "outputs", "henan_multi")
os.makedirs(OUT, exist_ok=True)

print("=" * 65)
print("🛰️  阶段 3C：河南省域 · 多指数遥感 GIS 管线")
print("   NDVI / EVI / SAVI / NDWI")
print("=" * 65)

# ====== 区域设置 ======
# 河南中南部 ~5市范围（周口+驻马店+信阳+漯河+平顶山）
# 约 150km × 280km，2-3 景 Sentinel-2 可覆盖
HENAN_BBOX = [113.50, 31.80, 115.50, 34.20]   # WGS84
# 可选：更小范围（周口周边）仅 1 景
# HENAN_BBOX = [114.50, 33.40, 115.00, 33.80]

print(f"\n📍 研究区域: [{HENAN_BBOX[0]:.2f}, {HENAN_BBOX[1]:.2f}, {HENAN_BBOX[2]:.2f}, {HENAN_BBOX[3]:.2f}]")

# ==============================
# Step 1: 河南县域边界 + 过滤
# ==============================
print("\n📥 Step 1: 河南县域边界...")

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
print(f"  ✅ {len(gdf)} 个行政区")

# 过滤到研究区域内
henan_box = box(*HENAN_BBOX)
mask = gdf.geometry.intersects(henan_box)
gdf_aoi = gdf[mask].copy()
print(f"  ✅ 研究区域内: {len(gdf_aoi)} 个")

gdf_aoi.to_file(os.path.join(OUT, "counties_aoi.geojson"), driver="GeoJSON")

# ==============================
# Step 2: STAC 搜索 + 逐景裁剪
# ==============================
print("\n🛰️  Step 2: STAC 搜索 + 逐景裁剪...")

from pystac_client import Client
import planetary_computer as pc

api = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
search = api.search(
    collections=["sentinel-2-l2a"],
    bbox=HENAN_BBOX,
    datetime="2024-04-01/2024-05-31",
    query={"eo:cloud_cover": {"lt": 15}},    # 低云要求！之前是 40，降到 15 减少景数
    sortby="-properties.eo:cloud_cover",
    limit=30,
)
items = [pc.sign(it) for it in search.items()]
print(f"  ✅ STAC 返回 {len(items)} 景（云量 < 15%）")

# 每个 mgrs_tile 只取云量最低的一景
tiles_seen = {}
for it in items:
    tile = it.properties.get("s2:mgrs_tile", "?")
    if tile not in tiles_seen:
        tiles_seen[tile] = it
selected = list(tiles_seen.values())

# footprint 过滤
aoi_poly = box(*HENAN_BBOX)
selected = [it for it in selected if shape(it.geometry).intersects(aoi_poly)]
print(f"  ✅ 去重+footprint 过滤后: {len(selected)} 景")
for i, it in enumerate(selected):
    tile = it.properties.get("s2:mgrs_tile", "?")
    cloud = it.properties.get("eo:cloud_cover", 0)
    dt = it.datetime.date()
    print(f"     [{i+1}] {tile} | {dt} | 云量 {cloud:.1f}%")

# 波段
BANDS = {"B02": "Blue", "B03": "Green", "B04": "Red", "B08": "NIR"}

# tmp 目录
TMP = os.path.join(OUT, "tmp")
os.makedirs(TMP, exist_ok=True)
for f in os.listdir(TMP):
    os.remove(os.path.join(TMP, f))

band_tiles = {b: [] for b in BANDS}   # 每个波段收集临时 tif 路径

for ti, it in enumerate(selected):
    tile = it.properties.get("s2:mgrs_tile", "?")
    print(f"\n  [{ti+1}/{len(selected)}] 🔪 裁剪 {tile} ...", flush=True)

    # 先开 B04 确认裁剪窗口
    try:
        with rasterio.open("/vsicurl/" + it.assets["B04"].href, timeout=120) as src:
            utm_bbox = transform_bounds("EPSG:4326", src.crs, *HENAN_BBOX)
            w_full = from_bounds(*utm_bbox, src.transform)

            # clamp 到影像边界内
            col_off = max(0, int(w_full.col_off))
            row_off = max(0, int(w_full.row_off))
            width = min(int(w_full.width), src.width - col_off)
            height = min(int(w_full.height), src.height - row_off)

            if width < 10 or height < 10:
                print(f"     ⏭️  裁剪区域太小 ({width}×{height})")
                continue

            w = Window(col_off, row_off, width, height)
            transform_clip = src.window_transform(w)
            crs_utm = src.crs
            print(f"     📐 窗口: {width}×{height} 像元")
    except Exception as e:
        print(f"     ⚠️  打开 B04 失败: {e}")
        continue

    # 读所有 4 个波段
    tile_ok = {}
    for band_key, band_name in BANDS.items():
        try:
            with rasterio.open("/vsicurl/" + it.assets[band_key].href, timeout=120) as src:
                arr = src.read(1, window=w).astype(np.float32) / 10000.0
                tile_ok[band_key] = arr
        except Exception as e:
            print(f"     ⚠️  {band_key} 失败: {e}")
            break

    if len(tile_ok) < 4:
        print(f"     ⏭️  波段不完整，跳过")
        continue

    # 存临时 tif
    for bk, arr in tile_ok.items():
        profile = {
            "driver": "GTiff", "height": arr.shape[0], "width": arr.shape[1],
            "count": 1, "dtype": "float32", "crs": crs_utm, "transform": transform_clip,
        }
        tmp = os.path.join(TMP, f"{tile}_{bk}.tif")
        with rasterio.open(tmp, "w", **profile) as dst:
            dst.write(arr, 1)
        band_tiles[bk].append(tmp)

    print(f"     ✅ 4 波段裁剪完成 → {width}×{height}")

print(f"\n  ✅ 逐景裁剪完成")
for b, paths in band_tiles.items():
    print(f"     {b} ({BANDS[b]}): {len(paths)} 块")

if not band_tiles["B04"]:
    print("\n❌ 没有裁剪成功！检查 bbox 或景覆盖")
    sys.exit(1)

# ==============================
# Step 3: merge + 多指数计算
# ==============================
print("\n🧮 Step 3: merge + 多指数...")

full_bands = {}
full_transform = None
full_crs = None

for bk, paths in band_tiles.items():
    if not paths:
        continue
    if len(paths) == 1:
        with rasterio.open(paths[0]) as src:
            full_bands[bk] = src.read(1)
            full_transform = src.transform
            full_crs = src.crs
    else:
        fps = [rasterio.open(p) for p in paths]
        merged, full_transform = merge(fps)
        full_bands[bk] = merged[0].astype(np.float32)
        full_crs = fps[0].crs
        for fp in fps:
            fp.close()
    print(f"  📎 {bk} merge: {full_bands[bk].shape}")

red = full_bands["B04"]
nir = full_bands["B08"]
blue = full_bands["B02"]
green = full_bands["B03"]
del full_bands; gc.collect()

# 多指数
ndvi = np.clip((nir - red) / (nir + red + 1e-8), -1, 1)
evi = np.clip(2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + 1e-8), -1, 1)
savi = np.clip((nir - red) / (nir + red + 0.5) * 1.5, -1, 1)
ndwi = np.clip((green - nir) / (green + nir + 1e-8), -1, 1)
del red, nir, blue, green; gc.collect()

for name, arr in [("NDVI", ndvi), ("EVI", evi), ("SAVI", savi), ("NDWI", ndwi)]:
    print(f"  🧮 {name}: 均值={np.nanmean(arr):.4f} | 范围=[{np.nanmin(arr):.3f}, {np.nanmax(arr):.3f}]")

# 写 GeoTIFF
profile_out = {
    "driver": "GTiff", "height": ndvi.shape[0], "width": ndvi.shape[1],
    "count": 1, "dtype": "float32", "crs": full_crs, "transform": full_transform,
    "BIGTIFF": "IF_SAFER",
}
for name, arr in [("ndvi", ndvi), ("evi", evi), ("savi", savi), ("ndwi", ndwi)]:
    tif = os.path.join(OUT, f"aoi_s2_{name}.tif")
    with rasterio.open(tif, "w", **profile_out) as dst:
        dst.write(arr, 1)
    print(f"  💾 aoi_s2_{name}.tif")

# ==============================
# Step 4: Zonal Statistics
# ==============================
print("\n📊 Step 4: Zonal Statistics...")
from rasterstats import zonal_stats

# 把 gdf 转到影像 CRS（UTM）做 zonal stats
gdf_proj = gdf_aoi.to_crs(full_crs)

indices = ["ndvi", "evi", "savi", "ndwi"]
stats_all = {}

for idx in indices:
    tif = os.path.join(OUT, f"aoi_s2_{idx}.tif")
    try:
        stats = zonal_stats(gdf_proj, tif,
                           stats=["mean", "std", "min", "max", "count", "percentile_90"],
                           all_touched=True, nodata=np.nan)
        stats_all[idx] = stats
        valid = sum(1 for s in stats if s["mean"] is not None and s["count"] > 0)
        print(f"  ✅ {idx.upper()}: {valid}/{len(stats)} 有数据")
    except Exception as e:
        print(f"  ❌ {idx} 失败: {e}")
        stats_all[idx] = [None] * len(gdf_proj)

# 合并到 GeoDataFrame
gdf_out = gdf_proj.reset_index(drop=True)
for idx in indices:
    for col in ["mean", "std", "min", "max", "count", "percentile_90"]:
        vals = [s[col] if s else np.nan for s in stats_all[idx]]
        gdf_out[f"{idx}_{col}"] = vals

gdf_out = gdf_out[gdf_out["ndvi_mean"].notna() & (gdf_out["ndvi_count"] > 0)]
print(f"  📋 {len(gdf_out)} 个行政区有完整数据")

# Top/Bottom
print("\n  🥇 NDVI Top 5:")
for _, r in gdf_out.sort_values("ndvi_mean", ascending=False).head(5).iterrows():
    print(f"     {r.get('name','?'):12s}  NDVI={r['ndvi_mean']:.3f}  EVI={r['evi_mean']:.3f}  SAVI={r['savi_mean']:.3f}")

print("\n  🔻 NDVI Bottom 5:")
for _, r in gdf_out.sort_values("ndvi_mean").head(5).iterrows():
    print(f"     {r.get('name','?'):12s}  NDVI={r['ndvi_mean']:.3f}  NDWI={r['ndwi_mean']:.3f}")

gdf_out.to_file(os.path.join(OUT, "counties_indices.shp"), driver="ESRI Shapefile")
print(f"  💾 counties_indices.shp")

# ==============================
# Step 5: 多指数专题图
# ==============================
print("\n🎨 Step 5: 专题图...")

fig, axes = plt.subplots(2, 2, figsize=(20, 16))

configs = [
    (ndvi, "ndvi_mean", "NDVI", "RdYlGn", -0.2, 0.8),
    (evi,  "evi_mean",  "EVI",  "RdYlGn", -0.2, 0.8),
    (savi, "savi_mean", "SAVI", "RdYlGn", -0.2, 0.8),
    (ndwi, "ndwi_mean", "NDWI", "Blues",  -0.5, 0.5),
]

for idx, (arr, col, title, cmap, vmin, vmax) in enumerate(configs):
    row, col_idx = idx // 2, idx % 2
    ax = axes[row, col_idx]

    step = max(1, arr.shape[0] // 800)
    small = arr[::step, ::step]
    im = ax.imshow(small, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(f"{title} · 县域均值着色", fontsize=12, fontweight="bold")
    ax.axis("off")

    gdf_out.plot(column=col, ax=ax, cmap=cmap,
                 edgecolor="white", linewidth=0.4,
                 vmin=vmin, vmax=vmax, alpha=0.7)

    plt.colorbar(im, ax=ax, shrink=0.6, label=title)

plt.suptitle("河南省 · 2024 小麦季 Sentinel-2 多指数专题图\n(中南部 5 市 · 多景裁剪 merge 拼接)",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
out_png = os.path.join(OUT, "henan_multi_index_map.png")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close()
print(f"  💾 {out_png}")

# 清理 tmp
import shutil
shutil.rmtree(TMP, ignore_errors=True)

# ==============================
# 指数均值对比表
# ==============================
print("\n📊 县域指数均值对比:")
print("-" * 55)
print(f"{'指数':<8} {'均值':>10} {'标准差':>10} {'最小':>10} {'最大':>10}")
print("-" * 55)
for idx in indices:
    m = np.nanmean([r[f"{idx}_mean"] for _, r in gdf_out.iterrows()])
    s = np.nanmean([r[f"{idx}_std"] for _, r in gdf_out.iterrows()])
    mn = np.nanmin([r[f"{idx}_min"] for _, r in gdf_out.iterrows()])
    mx = np.nanmax([r[f"{idx}_max"] for _, r in gdf_out.iterrows()])
    print(f"{idx.upper():<8} {m:>10.4f} {s:>10.4f} {mn:>10.4f} {mx:>10.4f}")

print("\n" + "=" * 65)
print(f"🎉 阶段 3C 管线完成！")
print(f"\n📦 产出 ({OUT}/):")
print(f"  aoi_s2_ndvi.tif / evi.tif / savi.tif / ndwi.tif")
print(f"  counties_indices.shp     县域 + 4 指数统计")
print(f"  counties_aoi.geojson     研究区域边界")
print(f"  henan_multi_index_map.png 🎨 2×2 专题图")
