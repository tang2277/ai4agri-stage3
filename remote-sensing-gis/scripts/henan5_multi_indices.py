"""
============================================================
阶段 3C+：河南中南部 5 市 · 多指数管线
============================================================
范围：周口+驻马店+信阳+漯河+平顶山
bbox: [113.00, 31.30, 115.50, 34.20] (~250km × ~320km)
多景裁剪 + merge + Zonal Stats + Folium 交互式地图
============================================================
"""
import os, sys, json, warnings, gc, time
warnings.filterwarnings("ignore")
os.environ["CURL_CA_BUNDLE"] = ""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds, Window
from rasterio.warp import transform_bounds
from rasterio.merge import merge
from shapely.geometry import shape, box
import requests

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "outputs", "henan5_multi")
os.makedirs(OUT, exist_ok=True)

# ===== 河南中南部 5 市 =====
HN5_BBOX = [113.00, 31.30, 115.50, 34.20]

print("=" * 65)
print("🛰️  河南中南部 5 市 · NDVI / EVI / SAVI / NDWI")
print("=" * 65)
print(f"📍 范围: [{HN5_BBOX[0]}, {HN5_BBOX[1]}, {HN5_BBOX[2]}, {HN5_BBOX[3]}]")

# ==============================
# Step 1: 县域边界
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

aoi_box = box(*HN5_BBOX)
gdf_aoi = gdf[gdf.geometry.intersects(aoi_box)].copy()
print(f"  ✅ {len(gdf)} 个行政区 → 研究区域内 {len(gdf_aoi)} 个")

gdf_aoi.to_file(os.path.join(OUT, "counties.geojson"), driver="GeoJSON")

# ==============================
# Step 2: STAC 搜索 + 多景裁剪
# ==============================
print("\n🛰️  Step 2: STAC 搜索...")
from pystac_client import Client
import planetary_computer as pc

api = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
search = api.search(
    collections=["sentinel-2-l2a"],
    bbox=HN5_BBOX,
    datetime="2024-04-01/2024-05-31",
    query={"eo:cloud_cover": {"lt": 20}},
    sortby="-properties.eo:cloud_cover",
    limit=30,
)
items = [pc.sign(it) for it in search.items()]
print(f"  ✅ 找到 {len(items)} 景（云量 < 20%）")

# 每个 tile 取最低云量
tiles_seen = {}
for it in items:
    tile = it.properties.get("s2:mgrs_tile", "?")
    if tile not in tiles_seen:
        tiles_seen[tile] = it
selected = list(tiles_seen.values())

# footprint 过滤
aoi_poly = box(*HN5_BBOX)
selected = [it for it in selected if shape(it.geometry).intersects(aoi_poly)]
print(f"  ✅ 过滤后 {len(selected)} 景真的覆盖研究区域")
for i, it in enumerate(selected):
    tile = it.properties.get("s2:mgrs_tile", "?")
    cloud = it.properties.get("eo:cloud_cover", 0)
    dt = it.datetime.date()
    print(f"     [{i+1}] {tile} | {dt} | 云量 {cloud:.1f}%")

# ===== 逐景裁剪 =====
BANDS = {"B02": "Blue", "B03": "Green", "B04": "Red", "B08": "NIR"}
TMP = os.path.join(OUT, "tmp")
os.makedirs(TMP, exist_ok=True)
for f in os.listdir(TMP):
    os.remove(os.path.join(TMP, f))

band_tiles = {b: [] for b in BANDS}

for ti, it in enumerate(selected):
    tile = it.properties.get("s2:mgrs_tile", "?")
    print(f"\n  [{ti+1}/{len(selected)}] 🔪 裁剪 {tile} ...")

    try:
        with rasterio.open("/vsicurl/" + it.assets["B04"].href, timeout=180) as src:
            utm_bbox = transform_bounds("EPSG:4326", src.crs, *HN5_BBOX)
            w_full = from_bounds(*utm_bbox, src.transform)

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
            print(f"     📐 窗口: {width}×{height}")
    except Exception as e:
        print(f"     ⚠️  打开 B04 失败: {e}")
        continue

    # 读所有波段
    tile_ok = {}
    all_ok = True
    for bk, bn in BANDS.items():
        t0 = time.time()
        try:
            with rasterio.open("/vsicurl/" + it.assets[bk].href, timeout=180) as src:
                arr = src.read(1, window=w).astype(np.float32) / 10000.0
                tile_ok[bk] = arr
                print(f"     ✅ {bk}: {arr.shape} in {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"     ⚠️  {bk} 失败: {e}")
            all_ok = False
            break

    if not all_ok:
        print(f"     ⏭️  此景跳过")
        continue

    # 存 tmp
    for bk, arr in tile_ok.items():
        profile = {
            "driver": "GTiff", "height": arr.shape[0], "width": arr.shape[1],
            "count": 1, "dtype": "float32", "crs": crs_utm, "transform": transform_clip,
        }
        tmp = os.path.join(TMP, f"{tile}_{bk}.tif")
        with rasterio.open(tmp, "w", **profile) as dst:
            dst.write(arr, 1)
        band_tiles[bk].append(tmp)

print(f"\n  ✅ 逐景裁剪完成")
for b, paths in band_tiles.items():
    print(f"     {b} ({BANDS[b]}): {len(paths)} 块")

if not band_tiles["B04"]:
    print("\n❌ 没有裁剪成功！")
    sys.exit(1)

# ==============================
# Step 3: merge + 多指数
# ==============================
print("\n🧮 Step 3: merge + 多指数...")

full_bands = {}
full_transform = None
full_crs = None

for bk, paths in band_tiles.items():
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
    print(f"  📎 {bk}: {full_bands[bk].shape}")

red = full_bands["B04"]
nir = full_bands["B08"]
blue = full_bands["B02"]
green = full_bands["B03"]
del full_bands; gc.collect()

ndvi = np.clip((nir - red) / (nir + red + 1e-8), -1, 1)
evi = np.clip(2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1 + 1e-8), -1, 1)
savi = np.clip((nir - red) / (nir + red + 0.5) * 1.5, -1, 1)
ndwi = np.clip((green - nir) / (green + nir + 1e-8), -1, 1)
del red, nir, blue, green; gc.collect()

for n, a in [("NDVI", ndvi), ("EVI", evi), ("SAVI", savi), ("NDWI", ndwi)]:
    print(f"  🧮 {n}: 均值={np.nanmean(a):.4f} | [{np.nanmin(a):.3f}, {np.nanmax(a):.3f}]")

# 写 tif
profile_out = {
    "driver": "GTiff", "height": ndvi.shape[0], "width": ndvi.shape[1],
    "count": 1, "dtype": "float32", "crs": full_crs, "transform": full_transform,
    "BIGTIFF": "IF_SAFER",
}
for n, a in [("ndvi", ndvi), ("evi", evi), ("savi", savi), ("ndwi", ndwi)]:
    tif = os.path.join(OUT, f"henan5_{n}.tif")
    with rasterio.open(tif, "w", **profile_out) as dst:
        dst.write(a, 1)
print(f"  💾 4 个 tif 已保存")

# ==============================
# Step 4: Zonal Statistics
# ==============================
print("\n📊 Step 4: Zonal Statistics...")
from rasterstats import zonal_stats

gdf_proj = gdf_aoi.to_crs(full_crs)

indices = ["ndvi", "evi", "savi", "ndwi"]
stats_all = {}

for idx in indices:
    tif = os.path.join(OUT, f"henan5_{idx}.tif")
    stats = zonal_stats(gdf_proj, tif,
                       stats=["mean", "std", "min", "max", "count"],
                       all_touched=True, nodata=np.nan)
    stats_all[idx] = stats
    valid = sum(1 for s in stats if s["mean"] is not None and s["count"] > 0)
    print(f"  ✅ {idx.upper()}: {valid}/{len(stats)} 有数据")

gdf_out = gdf_proj.reset_index(drop=True)
for idx in indices:
    for col in ["mean", "std", "min", "max", "count"]:
        vals = [s[col] if s else np.nan for s in stats_all[idx]]
        gdf_out[f"{idx}_{col}"] = vals

valid = gdf_out[gdf_out["ndvi_mean"].notna() & (gdf_out["ndvi_count"] > 0)]
print(f"  📋 {len(valid)} 个行政区有完整数据")

print("\n  🥇 NDVI Top 5:")
for _, r in valid.sort_values("ndvi_mean", ascending=False).head(5).iterrows():
    print(f"     {r.get('name','?'):12s}  NDVI={r['ndvi_mean']:.3f}  EVI={r['evi_mean']:.3f}")

print("\n  🔻 NDVI Bottom 5:")
for _, r in valid.sort_values("ndvi_mean").head(5).iterrows():
    print(f"     {r.get('name','?'):12s}  NDVI={r['ndvi_mean']:.3f}  NDWI={r['ndwi_mean']:.3f}")

valid.to_file(os.path.join(OUT, "henan5_indices.shp"), driver="ESRI Shapefile")

# ==============================
# Step 5: Folium 交互式地图
# ==============================
print("\n🗺️  Step 5: Folium 交互式地图...")
import folium
from folium.plugins import LayerControl, Fullscreen
from folium import GeoJsonTooltip

# 转回 WGS84
gdf_wgs = valid.to_crs("EPSG:4326")

center_lat = gdf_wgs.geometry.centroid.y.mean()
center_lon = gdf_wgs.geometry.centroid.x.mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=9,
    tiles="OpenStreetMap",
    control_scale=True,
)

INDEX_CONFIG = {
    "ndvi": ("NDVI 植被绿度", "YlGn", -0.2, 0.8),
    "evi":  ("EVI 增强植被", "YlGn", -0.2, 0.8),
    "savi": ("SAVI 土壤调节", "YlGn", -0.2, 0.6),
    "ndwi": ("NDWI 水体指数", "Blues", -0.5, 0.5),
}

for idx, (label, cmap, vmin, vmax) in INDEX_CONFIG.items():
    folium.Choropleth(
        geo_data=gdf_wgs,
        name=label,
        data=gdf_wgs,
        columns=["name", f"{idx}_mean"],
        key_on="feature.properties.name",
        fill_color=cmap,
        fill_opacity=0.7,
        line_opacity=0.5,
        legend_name=label,
        vmin=vmin, vmax=vmax,
        show=(idx == "ndvi"),
    ).add_to(m)

gjson = json.loads(gdf_wgs.to_json())
folium.GeoJson(
    gjson,
    name="县域边界",
    style_function=lambda x: {"fillOpacity": 0, "weight": 1, "color": "#333"},
    tooltip=GeoJsonTooltip(
        fields=["name"] + [f"{i}_mean" for i in indices] + ["ndvi_count"],
        aliases=["行政区"] + [f"{i.upper()} 均值" for i in indices] + ["像元数"],
        sticky=True,
    ),
    show=True,
).add_to(m)

LayerControl(collapsed=False).add_to(m)
Fullscreen().add_to(m)

html_path = os.path.join(OUT, "henan5_interactive_map.html")
m.save(html_path)
print(f"  💾 {html_path}")

# ==============================
# Step 6: 静态专题图
# ==============================
print("\n🎨 Step 6: 静态专题图...")
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

configs = [
    (ndvi, "ndvi_mean", "NDVI", "RdYlGn", -0.2, 0.8),
    (evi,  "evi_mean",  "EVI",  "RdYlGn", -0.2, 0.8),
    (savi, "savi_mean", "SAVI", "RdYlGn", -0.2, 0.6),
    (ndwi, "ndwi_mean", "NDWI", "Blues",  -0.5, 0.5),
]

for i, (arr, col, title, cmap, vmin, vmax) in enumerate(configs):
    r, c = i // 2, i % 2
    ax = axes[r, c]
    step = max(1, arr.shape[0] // 600)
    small = arr[::step, ::step]
    im = ax.imshow(small, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(f"{title} · 河南中南部 {len(gdf_aoi)} 县", fontsize=12, fontweight="bold")
    ax.axis("off")
    valid.plot(column=col, ax=ax, cmap=cmap, edgecolor="white", linewidth=0.4,
               vmin=vmin, vmax=vmax, alpha=0.7)
    plt.colorbar(im, ax=ax, shrink=0.6, label=title)

plt.suptitle("河南中南部 5 市 · 2024 小麦季 Sentinel-2 多指数专题图", fontsize=14, fontweight="bold")
plt.tight_layout()
static_png = os.path.join(OUT, "henan5_multi_map.png")
plt.savefig(static_png, dpi=150, bbox_inches="tight")
plt.close()
print(f"  💾 {static_png}")

# 清理 tmp
import shutil
shutil.rmtree(TMP, ignore_errors=True)

# ==============================
# 完成
# ==============================
print("\n" + "=" * 65)
print("🎉 河南中南部 5 市 · 多指数管线完成！")
print("=" * 65)
print(f"""
📦 产出 ({OUT}/):
  栅格: henan5_ndvi.tif / evi.tif / savi.tif / ndwi.tif
  矢量: henan5_indices.shp     {len(valid)} 个县域 + 4 指数统计
  交互: henan5_interactive_map.html  🗺️ Folium 可缩放点击地图
  静态: henan5_multi_map.png   🎨 2×2 专题图
""")

print("📊 县域指数均值对比:")
print("-" * 55)
print(f"{'指数':<8} {'均值':>10} {'标准差':>10} {'最小':>10} {'最大':>10}")
print("-" * 55)
for idx in indices:
    m = np.nanmean([r[f"{idx}_mean"] for _, r in valid.iterrows()])
    s = np.nanmean([r[f"{idx}_std"] for _, r in valid.iterrows()])
    mn = np.nanmin([r[f"{idx}_min"] for _, r in valid.iterrows()])
    mx = np.nanmax([r[f"{idx}_max"] for _, r in valid.iterrows()])
    print(f"{idx.upper():<8} {m:>10.4f} {s:>10.4f} {mn:>10.4f} {mx:>10.4f}")
