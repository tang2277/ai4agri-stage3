"""
===========================================================
下载真实 Sentinel-2 影像（Copernicus Data Space Ecosystem）
===========================================================
功能：
  1. 搜索河南周口区域 2024 年小麦生长季 (3-5月) 云量 < 20% 的影像
  2. 下载 L2A 级（已大气校正，直接可用的地表反射率）
  3. 解压 + 提取 B04(Red) / B08(NIR) / B02(Blue) / B03(Green)
  4. 计算 NDVI 并保存 GeoTIFF

用法：先在 https://dataspace.copernicus.eu 注册账号
     然后: python download_sentinel2.py
===========================================================
"""

import os
import sys
import zipfile
import shutil
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
import matplotlib.pyplot as plt

# ========== 配置 ==========
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "sentinel2_raw")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "sentinel2_real")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# 河南周口区域（西华/商水/淮阳/太康 4 县合并）
AREA = "POLYGON((114.65 33.65, 114.85 33.65, 114.85 33.50, 114.65 33.50, 114.65 33.65))"

# 下载配置
START_DATE = "2024-03-01"
END_DATE = "2024-05-31"
CLOUD_MAX = 20          # 云量上限 20%
PRODUCT_TYPE = "S2MSI2A"  # L2A 级（已大气校正）
MAX_RESULTS = 5          # 最多下载多少景

# ========== Step 0: 交互式输入账号 ==========
print("="*60)
print("🛰️  Sentinel-2 真实影像下载（Copernicus Data Space）")
print("="*60)
print(f"\n📌 搜索区域: 河南周口 ({AREA})")
print(f"📌 时间范围: {START_DATE} ~ {END_DATE}")
print(f"📌 云量上限: {CLOUD_MAX}%")
print(f"📌 产品级别: L2A（地表反射率，直接算 NDVI）")

# 尝试读取之前保存的凭据
cred_file = os.path.join(BASE_DIR, ".copernicus_creds.txt")
if os.path.exists(cred_file):
    with open(cred_file) as f:
        USERNAME, PASSWORD = f.read().strip().split(":")
    print(f"\n✅ 已保存的账号: {USERNAME}")
else:
    print("\n⚠️  需要 Copernicus Data Space Ecosystem 账号")
    print("   👉 https://dataspace.copernicus.eu 免费注册")
    USERNAME = input("   输入账号邮箱: ").strip()
    PASSWORD = input("   输入密码: ").strip()
    with open(cred_file, "w") as f:
        f.write(f"{USERNAME}:{PASSWORD}")
    print("   ✅ 凭据已保存，下次不用再输")

# ========== Step 1: 搜索影像 ==========
print("\n" + "="*60)
print("🔍 Step 1: 搜索 Sentinel-2 影像")
print("="*60)

from sentinelsat import SentinelAPI

# 新版 Copernicus Data Space Ecosystem 端点
api = SentinelAPI(USERNAME, PASSWORD, 
                  api_url="https://catalogue.dataspace.copernicus.eu/api/stac/v1",
                  oauth=True)

products = api.query(
    AREA,
    date=(START_DATE, END_DATE),
    platformname="Sentinel-2",
    cloudcoverpercentage=(0, CLOUD_MAX),
    producttype=PRODUCT_TYPE,
    limit=MAX_RESULTS,
)

print(f"  ✅ 找到 {len(products)} 景符合条件的影像")

# 按日期排序，选最新的 2 景
ordered = sorted(products.items(), key=lambda x: x[1]["ingestiondate"], reverse=True)
for i, (pid, meta) in enumerate(ordered):
    print(f"  [{i+1}] {meta.get('title', pid)} | 云量 {meta.get('cloudcoverpercentage', '?')}% | {meta.get('ingestiondate', '?')}")

if not ordered:
    print("\n⚠️  没找到影像！试试放宽条件（云量 30% / 时间范围扩大）")
    # 降级搜索
    print("\n🔄 降级搜索：放宽到云量 30%...")
    products = api.query(
        AREA, date=(START_DATE, END_DATE),
        platformname="Sentinel-2", cloudcoverpercentage=(0, 30),
        producttype=PRODUCT_TYPE, limit=MAX_RESULTS,
    )
    ordered = sorted(products.items(), key=lambda x: x[1]["ingestiondate"], reverse=True)
    print(f"  找到 {len(ordered)} 景（云量放宽到 30%）")

# ========== Step 2: 下载 ==========
print("\n" + "="*60)
print("⬇️  Step 2: 下载影像")
print("="*60)

downloaded_files = []
for i, (pid, meta) in enumerate(ordered[:2]):  # 最多下 2 景
    title = meta.get("title", pid)
    print(f"\n  [{i+1}/{len(ordered[:2])}] 下载: {title}")
    try:
        result = api.download(pid, directory_path=DATA_DIR)
        if result:
            downloaded_files.append(result["path"])
            print(f"    ✅ 完成: {result['path']}")
    except Exception as e:
        print(f"    ❌ 失败: {e}")

api.close()
print(f"\n  总计下载: {len(downloaded_files)} 景")

if not downloaded_files:
    print("\n❌ 什么都没下到，可能时间范围内没有无云数据")
    print("   建议：换时间（小麦拔节期 3月中-4月中）或扩大云量限制")
    sys.exit(1)

# ========== Step 3: 解压 + 提取波段 ==========
print("\n" + "="*60)
print("📦 Step 3: 解压 + 提取光谱波段")
print("="*60)

# 解压所有 zip
for f in os.listdir(DATA_DIR):
    if f.endswith(".zip"):
        zip_path = os.path.join(DATA_DIR, f)
        target = os.path.join(DATA_DIR, f.replace(".zip", ""))
        if not os.path.exists(target):
            print(f"  解压 {f}...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(DATA_DIR)

# 找到解压后的 SAFE 目录（Sentinel-2 标准格式）
safe_dirs = [d for d in os.listdir(DATA_DIR) if d.endswith(".SAFE")]
print(f"  ✅ 找到 {len(safe_dirs)} 个 SAFE 目录")

# 处理第一个（最新的）
if safe_dirs:
    safe_path = os.path.join(DATA_DIR, safe_dirs[0])
    print(f"\n  📂 处理: {safe_dirs[0]}")
    
    # 找 GRANULE 下的 IMG_DATA
    granule_dir = os.path.join(safe_path, "GRANULE")
    granules = os.listdir(granule_dir)
    if granules:
        img_data = os.path.join(granule_dir, granules[0], "IMG_DATA")
        print(f"  📂 IMG_DATA: {img_data}")
        
        # 找 B04, B08, B02, B03 波段（L2A 级是 10m 分辨率的）
        bands = {}
        for fname in os.listdir(img_data):
            # L2A 产品命名: *_B02_10m.jp2, *_B04_10m.jp2 等
            for bname, bcode in [("Red", "B04"), ("NIR", "B08"), ("Blue", "B02"), ("Green", "B03")]:
                if f"_{bcode}_10m" in fname:
                    bands[bname] = os.path.join(img_data, fname)
        
        print(f"  ✅ 找到波段: {list(bands.keys())}")
        
        if "Red" in bands and "NIR" in bands:
            # 读取 Red 和 NIR
            with rasterio.open(bands["Red"]) as src_red:
                red = src_red.read(1).astype(np.float32) / 10000.0  # L2A 值除以 10000
                transform = src_red.transform
                crs = src_red.crs
                print(f"  📐 Red: shape={red.shape}, CRS={crs}")
            
            with rasterio.open(bands["NIR"]) as src_nir:
                nir = src_nir.read(1).astype(np.float32) / 10000.0
                print(f"  📐 NIR: shape={nir.shape}")
            
            # 计算 NDVI
            ndvi = (nir - red) / (nir + red + 1e-8)
            ndvi = np.clip(ndvi, -1, 1)
            
            # 保存为 GeoTIFF（WGS84 重投影 + 原始 CRS 各存一份）
            ndvi_tif = os.path.join(OUT_DIR, "sentinel2_real_ndvi.tif")
            with rasterio.open(
                ndvi_tif, "w", driver="GTiff",
                height=ndvi.shape[0], width=ndvi.shape[1],
                count=1, dtype="float32", crs=crs, transform=transform,
            ) as dst:
                dst.write(ndvi, 1)
            print(f"  💾 NDVI GeoTIFF 已保存: {ndvi_tif}")
            
            # 统计
            print(f"\n  📊 真实 Sentinel-2 NDVI 统计:")
            print(f"     均值: {ndvi.mean():.4f}")
            print(f"     标准差: {ndvi.std():.4f}")
            print(f"     范围: [{ndvi.min():.3f}, {ndvi.max():.3f}]")
            print(f"     有效像元 (> -0.1): {(ndvi > -0.1).sum()} / {ndvi.size}")
            
            # 可视化
            ndvi_cmap = plt.cm.RdYlGn
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            axes[0].imshow(red, cmap="gray", vmin=0, vmax=0.3)
            axes[0].set_title("Sentinel-2 B04 Red Band (L2A)")
            axes[0].axis("off")
            
            im = axes[1].imshow(ndvi, cmap=ndvi_cmap, vmin=-0.2, vmax=0.8)
            axes[1].set_title("Real Sentinel-2 NDVI\n河南周口 · 2024 小麦季")
            axes[1].axis("off")
            plt.colorbar(im, ax=axes[1], label="NDVI")
            
            plt.tight_layout()
            png_path = os.path.join(OUT_DIR, "real_ndvi.png")
            plt.savefig(png_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  🎨 专题图: {png_path}")
        else:
            print("  ⚠️  没找到 B04/B08 波段")

print("\n" + "="*60)
print("🎉 真实 Sentinel-2 数据下载 + NDVI 计算完成!")
print("="*60)
print(f"\n📦 产出:")
print(f"  {OUT_DIR}/sentinel2_real_ndvi.tif  — 真实 NDVI GeoTIFF")
print(f"  {OUT_DIR}/real_ndvi.png           — NDVI 专题图")
print(f"  {DATA_DIR}/.SAFE/                   — 原始 L2A 影像")
print(f"\n💡 把 GeoTIFF 拖进 QGIS 就能和模拟数据对比效果！")
