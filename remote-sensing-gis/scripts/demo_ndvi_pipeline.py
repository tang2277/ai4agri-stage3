"""
===========================================================
阶段 3B · 遥感 + GIS 入门 Demo
===========================================================
功能：模拟 Sentinel-2 影像 → 计算 NDVI → 可视化专题图
技术栈：rasterio + matplotlib + numpy
时间：约 2 分钟（纯 CPU）
===========================================================

作物学背景解读：
- NDVI = (NIR - Red) / (NIR + Red)，范围 [-1, 1]
- NDVI > 0.5 → 作物长势好（深绿色）
- NDVI 0.2-0.5 → 长势一般（浅绿色）
- NDVI < 0.2 → 裸土/岩石（黄色）
- NDVI < 0 → 水体/云（蓝色）
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import rasterio
from rasterio.transform import from_bounds
import os

# ========== 1. 生成模拟 Sentinel-2 影像 ==========
print("🔄 Step 1: 生成模拟 Sentinel-2 多光谱影像...")

# 模拟 100x100 像素的地块（分辨率 10m → 实际 1km×1km）
H, W = 100, 100
np.random.seed(42)

# 生成地块边界（模拟 4 个不同作物地块）
field = np.zeros((H, W), dtype=np.uint8)
field[20:45, 10:50] = 1   # 地块 A: 小麦（长势好）
field[20:50, 55:90] = 2   # 地块 B: 玉米（长势中等）
field[55:80, 15:55] = 3   # 地块 C: 大豆（长势差）
field[55:85, 60:90] = 4   # 地块 D: 裸土

def gen_spectral(layer_id, noise=0.05):
    """模拟不同地物的光谱特征"""
    profiles = {
        0: (0.05, 0.15),   # 背景/道路: Red=0.05, NIR=0.15
        1: (0.10, 0.75),   # 小麦:      Red=0.10, NIR=0.75 → NDVI ≈ 0.76
        2: (0.18, 0.55),   # 玉米:      Red=0.18, NIR=0.55 → NDVI ≈ 0.51
        3: (0.25, 0.35),   # 大豆:      Red=0.25, NIR=0.35 → NDVI ≈ 0.17
        4: (0.35, 0.40),   # 裸土:      Red=0.35, NIR=0.40 → NDVI ≈ 0.07
    }
    red_val, nir_val = profiles[layer_id]
    red = np.random.normal(red_val, noise, (H, W)) * (field == layer_id)
    nir = np.random.normal(nir_val, noise, (H, W)) * (field == layer_id)
    return red, nir

red_band = np.zeros((H, W))
nir_band = np.zeros((H, W))
for lid in range(5):
    r, n = gen_spectral(lid)
    red_band += r
    nir_band += n

# 归一化到 [0, 1]
red_band = np.clip(red_band, 0, 1)
nir_band = np.clip(nir_band, 0, 1)

print(f"  ✅ Red band:  shape={red_band.shape}, range=[{red_band.min():.3f}, {red_band.max():.3f}]")
print(f"  ✅ NIR band:  shape={nir_band.shape}, range=[{nir_band.min():.3f}, {nir_band.max():.3f}]")

# ========== 2. 保存为 GeoTIFF（带地理坐标）==========
print("\n💾 Step 2: 保存为 GeoTIFF...")
os.makedirs("outputs", exist_ok=True)

# 模拟地理位置：河南周口附近（东经 114.7°, 北纬 33.6°）
transform = from_bounds(114.70, 33.58, 114.71, 33.57, W, H)
crs = "EPSG:4326"  # WGS84

for band_data, band_name in [(red_band, "B04_red"), (nir_band, "B08_nir")]:
    with rasterio.open(
        f"outputs/sentinel2_{band_name}.tif",
        "w", driver="GTiff", height=H, width=W,
        count=1, dtype="float32", crs=crs, transform=transform,
    ) as dst:
        dst.write(band_data.astype(np.float32), 1)

print("  ✅ GeoTIFF 已保存: outputs/sentinel2_B04_red.tif")
print("  ✅ GeoTIFF 已保存: outputs/sentinel2_B08_nir.tif")

# ========== 3. 读取 GeoTIFF + 计算 NDVI ==========
print("\n🧮 Step 3: 计算 NDVI...")

with rasterio.open("outputs/sentinel2_B04_red.tif") as src:
    red = src.read(1)
    print(f"  📖 Red band 读取成功, CRS={src.crs}")

with rasterio.open("outputs/sentinel2_B08_nir.tif") as src:
    nir = src.read(1)
    print(f"  📖 NIR band 读取成功, CRS={src.crs}")

# NDVI 公式
ndvi = (nir - red) / (nir + red + 1e-8)  # 加小量避免除零
ndvi = np.clip(ndvi, -1, 1)

print(f"  ✅ NDVI 计算完成: 均值={ndvi.mean():.3f}, 范围=[{ndvi.min():.3f}, {ndvi.max():.3f}]")

# ========== 4. 可视化专题图 ==========
print("\n🎨 Step 4: 生成作物长势专题图...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle("Sentinel-2 遥感影像 · NDVI 作物长势分析\n(模拟数据 · 河南周口 · 2024 小麦季)", 
             fontsize=14, fontweight="bold")

# 4-1. Red 波段
axes[0, 0].imshow(red, cmap="gray", vmin=0, vmax=0.6)
axes[0, 0].set_title("Red 波段 (B04)")
axes[0, 0].axis("off")
plt.colorbar(axes[0, 0].imshow(red, cmap="gray", vmin=0, vmax=0.6), ax=axes[0, 0], shrink=0.8)

# 4-2. NIR 波段
axes[0, 1].imshow(nir, cmap="gray", vmin=0, vmax=0.8)
axes[0, 1].set_title("NIR 波段 (B08)")
axes[0, 1].axis("off")
plt.colorbar(axes[0, 1].imshow(nir, cmap="gray", vmin=0, vmax=0.8), ax=axes[0, 1], shrink=0.8)

# 4-3. 假彩色合成（NIR, Red, Green）→ 植被红色
rgb = np.stack([nir, red, red * 0.5], axis=-1)
rgb = np.clip(rgb / rgb.max(), 0, 1)
axes[1, 0].imshow(rgb)
axes[1, 0].set_title("假彩色合成 (NIR-Red-Green)")
axes[1, 0].axis("off")

# 4-4. NDVI 专题图（核心产出！）
ndvi_cmap = LinearSegmentedColormap.from_list("ndvi", [
    "#436A02", "#7CB518", "#F5C518", "#ED8B00", "#C41E3A", "#006BA4"
])
im = axes[1, 1].imshow(ndvi, cmap=ndvi_cmap, vmin=-0.2, vmax=0.8)
axes[1, 1].set_title("NDVI 作物长势专题图")
axes[1, 1].axis("off")
cb = plt.colorbar(im, ax=axes[1, 1], shrink=0.8, ticks=[-0.2, 0, 0.2, 0.4, 0.6, 0.8])
cb.ax.set_yticklabels(["水体/云", "裸土", "长势差", "一般", "良好", "优秀"])

plt.tight_layout()
plt.savefig("outputs/ndvi_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ 专题图已保存: outputs/ndvi_analysis.png")

# ========== 5. 统计分析 + 作物学解读 ==========
print("\n📊 Step 5: 作物长势统计分析")

field_names = {0: "背景/道路", 1: "小麦", 2: "玉米", 3: "大豆", 4: "裸土"}
print(f"\n{'地块':<10} {'像元数':<10} {'平均 NDVI':<12} {'长势评价'}")
print("-" * 50)

for fid, fname in field_names.items():
    mask = field == fid
    if mask.sum() > 0:
        mean_ndvi = ndvi[mask].mean()
        if mean_ndvi > 0.5:
            status = "✅ 长势良好"
        elif mean_ndvi > 0.2:
            status = "⚠️ 长势一般"
        elif mean_ndvi > 0:
            status = "❌ 长势差"
        else:
            status = "— 非植被"
        print(f"{fname:<10} {mask.sum():<10} {mean_ndvi:<12.3f} {status}")

print(f"\n{'='*50}")
print(f"🎯 阶段 3B Demo 完成！")
print(f"   产出: outputs/ndvi_analysis.png (专题图)")
print(f"         outputs/sentinel2_*.tif (GeoTIFF 数据)")
print(f"{'='*50}")
print("\n📚 下一步：")
print("   1. 安装 QGIS → 打开 GeoTIFF → 可视化体验")
print("   2. Google Earth Engine → 下载真实 Sentinel-2 影像")
print("   3. 导入真实地块边界 → 制作县/镇级专题图")
