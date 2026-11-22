"""
============================================================
阶段 3C+：周口 · 春 vs 秋 NDVI 时序变化
============================================================
同一块区域，两个时间点：
  春季 (2024-04) → 冬小麦灌浆期，NDVI 应该高
  秋季 (2024-10) → 秋收后休耕地，NDVI 应该低
NDVI 差值 = 春 - 秋 → 正值=春高，负值=秋高
============================================================
"""
import os, json, warnings, gc, time
warnings.filterwarnings("ignore")
os.environ["CURL_CA_BUNDLE"] = ""

import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds, Window
from rasterio.warp import transform_bounds
from shapely.geometry import shape, box
import requests

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "outputs", "zhoukou_timeseries")
os.makedirs(OUT, exist_ok=True)

ZK_BBOX = [114.65, 33.50, 114.85, 33.65]   # 已验证稳定

print("=" * 65)
print("📈 周口 · 春 vs 秋 NDVI 时序变化")
print("=" * 65)

from pystac_client import Client
import planetary_computer as pc

api = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

# ===== 两个时间点 =====
TIME_POINTS = {
    "spring": ("2024-04-01/2024-04-30", "灌浆期"),
    "autumn": ("2024-10-01/2024-10-31", "休耕期"),
}

results = {}

for season, (dt_range, label) in TIME_POINTS.items():
    print(f"\n{'='*40}")
    print(f"🌱 {season} ({dt_range}, {label})")
    print(f"{'='*40}")

    search = api.search(
        collections=["sentinel-2-l2a"],
        bbox=ZK_BBOX,
        datetime=dt_range,
        query={"eo:cloud_cover": {"lt": 20}},
        sortby="-properties.eo:cloud_cover",
        limit=5,
    )
    items = [pc.sign(it) for it in search.items()]
    print(f"  ✅ 找到 {len(items)} 景（云量 < 20%）")

    if not items:
        print(f"  ⚠️  这个时间点没找到！")
        continue

    it = items[0]
    tile = it.properties.get("s2:mgrs_tile", "?")
    cc = it.properties.get("eo:cloud_cover", 0)
    dt = it.datetime.date()
    print(f"  🎯 选中: {tile} | {dt} | 云量 {cc:.1f}%")

    # 裁剪
    print(f"\n  🔪 裁剪 4 波段...")
    BANDS = {"B02": "Blue", "B03": "Green", "B04": "Red", "B08": "NIR"}
    arrays = {}

    # 先开 B04 算裁剪窗口，所有波段复用
    with rasterio.open("/vsicurl/" + it.assets["B04"].href, timeout=120) as src_ref:
        utm_bbox = transform_bounds("EPSG:4326", src_ref.crs, *ZK_BBOX)
        w = from_bounds(*utm_bbox, src_ref.transform)
        w_clamped = Window(
            max(0, int(w.col_off)),
            max(0, int(w.row_off)),
            min(int(w.width), src_ref.width - max(0, int(w.col_off))),
            min(int(w.height), src_ref.height - max(0, int(w.row_off))),
        )
        transform_clip = src_ref.window_transform(w_clamped)
        crs_utm = src_ref.crs

    for bk, bn in BANDS.items():
        t0 = time.time()
        url = "/vsicurl/" + it.assets[bk].href
        with rasterio.open(url, timeout=120) as src:
            arr = src.read(1, window=w_clamped).astype('float32') / 10000.0
        arrays[bk] = arr
        print(f"     ✅ {bk}: {arr.shape} in {time.time()-t0:.1f}s")

    # NDVI
    ndvi = np.clip(
        (arrays["B08"] - arrays["B04"]) / (arrays["B08"] + arrays["B04"] + 1e-8),
        -1, 1
    )

    results[season] = {
        "ndvi": ndvi,
        "date": dt,
        "cloud": cc,
        "label": label,
        "tile": tile,
        "transform": transform_clip,
        "crs": crs_utm,
    }

    print(f"\n  📊 NDVI 均值: {np.nanmean(ndvi):.4f} | 范围 [{np.nanmin(ndvi):.3f}, {np.nanmax(ndvi):.3f}]")

    # 写 tif
    profile = {
        "driver": "GTiff", "height": ndvi.shape[0], "width": ndvi.shape[1],
        "count": 1, "dtype": "float32", "crs": crs_utm, "transform": transform_clip,
    }
    with rasterio.open(os.path.join(OUT, f"zk_{season}_ndvi.tif"), "w", **profile) as dst:
        dst.write(ndvi, 1)

# ===== 差值 =====
if "spring" in results and "autumn" in results:
    print(f"\n{'='*40}")
    print("📊 春 vs 秋 对比")
    print(f"{'='*40}")

    ndvi_spring = results["spring"]["ndvi"]
    ndvi_autumn = results["autumn"]["ndvi"]

    # 确保 shape 相同
    min_h = min(ndvi_spring.shape[0], ndvi_autumn.shape[0])
    min_w = min(ndvi_spring.shape[1], ndvi_autumn.shape[1])
    ndvi_spring = ndvi_spring[:min_h, :min_w]
    ndvi_autumn = ndvi_autumn[:min_h, :min_w]

    diff = ndvi_spring - ndvi_autumn   # 正值=春>秋
    print(f"\n  春季均值: {np.nanmean(ndvi_spring):.4f}")
    print(f"  秋季均值: {np.nanmean(ndvi_autumn):.4f}")
    print(f"  差值均值: {np.nanmean(diff):.4f}")
    print(f"  差值范围: [{np.nanmin(diff):.3f}, {np.nanmax(diff):.3f}]")

    # ===== 专题图 =====
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 春季
    im0 = axes[0].imshow(ndvi_spring, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
    axes[0].set_title(f"春季 NDVI\n{results['spring']['date']} · {results['spring']['label']}", fontsize=11)
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], shrink=0.7)

    # 秋季
    im1 = axes[1].imshow(ndvi_autumn, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
    axes[1].set_title(f"秋季 NDVI\n{results['autumn']['date']} · {results['autumn']['label']}", fontsize=11)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], shrink=0.7)

    # 差值（红=春低/秋高，绿=春高/秋低）
    im2 = axes[2].imshow(diff, cmap="RdYlGn", vmin=-0.3, vmax=0.5)
    axes[2].set_title(f"差值 (春-秋)\n正值=春更高（小麦季特征）", fontsize=11)
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], shrink=0.7, label="ΔNDVI")

    plt.suptitle("周口区域 · Sentinel-2 NDVI 时序变化\n冬小麦灌浆期 vs 休耕期", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "zk_timeseries_ndvi.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  💾 专题图: zk_timeseries_ndvi.png")

    # 差值 tif
    profile2 = {
        "driver": "GTiff", "height": diff.shape[0], "width": diff.shape[1],
        "count": 1, "dtype": "float32", "crs": results["spring"]["crs"],
        "transform": results["spring"]["transform"],
    }
    with rasterio.open(os.path.join(OUT, "zk_spring_minus_autumn.tif"), "w", **profile2) as dst:
        dst.write(diff, 1)

    # ===== 分布直方图 =====
    fig, ax = plt.subplots(figsize=(10, 5))
    valid = ~np.isnan(diff)
    ax.hist(ndvi_spring[valid], bins=50, alpha=0.5, label=f"春季 ({results['spring']['date']})", color="green")
    ax.hist(ndvi_autumn[valid], bins=50, alpha=0.5, label=f"秋季 ({results['autumn']['date']})", color="red")
    ax.set_xlabel("NDVI")
    ax.set_ylabel("像元数")
    ax.set_title("周口 NDVI 分布：春 vs 秋")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "zk_ndvi_histogram.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 直方图: zk_ndvi_histogram.png")

print(f"\n📦 产出 ({OUT}/):")
print(f"  zk_spring_ndvi.tif / zk_autumn_ndvi.tif")
print(f"  zk_spring_minus_autumn.tif")
print(f"  zk_timeseries_ndvi.png     🎨 三联图")
print(f"  zk_ndvi_histogram.png      📊 分布对比")
print("\n🎉 时序变化完成！")
