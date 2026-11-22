"""
============================================================
零注册下载真实 Sentinel-2 并算 NDVI（Planetary Computer）
============================================================
使用 rasterio 远程窗口读取 COG → 不需要下载整景！
完全零注册，微软公开 Planetary Computer STAC。
============================================================
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# 修复 Windows PROJ 数据库路径
os.environ["PROJ_LIB"] = r"C:\Users\Leng\miniconda3\envs\ai4agri\lib\site-packages\pyproj\proj_dir\share\proj"
os.environ["PROJ_DATA"] = os.environ["PROJ_LIB"]

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "outputs", "sentinel2_real")
os.makedirs(OUT, exist_ok=True)

print("="*60)
print("🛰️  Planetary Computer: Sentinel-2 L2A → 真实 NDVI")
print("="*60)

# ========== Step 1: 搜 ==========
print("\n🔍 Step 1: 搜索...")
from pystac_client import Client
import planetary_computer as pc

bbox = [114.65, 33.50, 114.85, 33.65]  # 河南周口

api = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
search = api.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime="2024-03-01/2024-05-31",
    query={"eo:cloud_cover": {"lt": 30}},
    sortby="-properties.eo:cloud_cover",
    limit=5,
)
items = [pc.sign(it) for it in search.items()]
print(f"  ✅ 找到 {len(items)} 景")

for i, it in enumerate(items[:3]):
    print(f"  [{i+1}] {it.datetime.date()} | 云量 {it.properties.get('eo:cloud_cover',0):.1f}%")

# ========== Step 2: 远程窗口读取 COG ==========
print("\n📦 Step 2: 远程裁剪 + 读取 B04/B08 ...")
import rasterio
from rasterio.windows import from_bounds

item = items[0]  # 云量最低的

# 用 /vsicurl/ + 签名 URL 直接远程读 COG
red_url = "/vsicurl/" + item.assets["B04"].href
nir_url = "/vsicurl/" + item.assets["B08"].href

os.environ["CURL_CA_BUNDLE"] = ""  # Windows 下避免 SSL 问题

with rasterio.open(red_url) as src_r:
    print(f"  Red CRS: {src_r.crs} | shape: {src_r.shape} | 分辨率: {src_r.res}")
    # bbox 在 WGS84，影像在 UTM，先转
    from rasterio.warp import transform_bounds
    utm_bbox = transform_bounds("EPSG:4326", src_r.crs, *bbox)
    w = from_bounds(*utm_bbox, src_r.transform)
    red = src_r.read(1, window=w).astype(np.float32) / 10000.0
    transform = src_r.window_transform(w)
    crs = src_r.crs
    print(f"  ✅ Red 裁剪后: {red.shape}")

with rasterio.open(nir_url) as src_n:
    nir = src_n.read(1, window=w).astype(np.float32) / 10000.0
    print(f"  ✅ NIR 裁剪后: {nir.shape}")

# ========== Step 3: 计算 NDVI ==========
ndvi = (nir - red) / (nir + red + 1e-8)
ndvi = np.clip(ndvi, -1, 1)
print(f"\n🧮 NDVI shape: {ndvi.shape}")
print(f"   均值: {np.nanmean(ndvi):.4f}")
print(f"   范围: [{np.nanmin(ndvi):.3f}, {np.nanmax(ndvi):.3f}]")

# ========== Step 4: 保存 ==========
print("\n💾 Step 4: 保存 GeoTIFF + PNG ...")
with rasterio.open(
    os.path.join(OUT, "real_s2_ndvi.tif"), "w",
    driver="GTiff", height=ndvi.shape[0], width=ndvi.shape[1],
    count=1, dtype="float32", crs=crs, transform=transform,
) as dst:
    dst.write(ndvi, 1)
print(f"  ✅ real_s2_ndvi.tif (CRS={crs}, {ndvi.shape[1]}×{ndvi.shape[0]} 像素)")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].imshow(red, cmap="gray", vmin=0, vmax=0.3)
axes[0].set_title(f"Sentinel-2 B04 Red\n{item.datetime.date()}")
axes[0].axis("off")
im = axes[1].imshow(ndvi, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
axes[1].set_title(f"Real Sentinel-2 NDVI\n河南周口 · 2024 小麦季")
axes[1].axis("off")
plt.colorbar(im, ax=axes[1], label="NDVI")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "real_s2_ndvi.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✅ real_s2_ndvi.png")

# ========== 完成 ==========
print(f"\n🎉 真实 Sentinel-2 NDVI 搞定!")
print(f"   GeoTIFF → {OUT}/real_s2_ndvi.tif")
print(f"   PNG     → {OUT}/real_s2_ndvi.png")
print(f"\n💡 拖进 QGIS 就能和模拟数据对比！")
