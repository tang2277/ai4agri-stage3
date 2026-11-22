# AI for Agriculture · Stage 3

> YOLOv8 作物病害检测 + Sentinel-2 遥感 GIS 管线 + 多指数 + 时序变化

---

## 🏗️ 阶段 3A：YOLOv8 目标检测 ✅

**仓库子目录**：`crop-detect-yolov8/`

使用 **YOLOv8-Nano** 目标检测模型，在 **PlantDoc** 数据集上训练 29 类作物病害检测器。CPU 快速验证 + AutoDL GPU 云 20 epoch 正式训练，**mAP50 = 62.05%**。

| 指标 | 值 |
|------|-----|
| **mAP50** | **62.05%** |
| mAP50-95 | 48.63% |
| 训练耗时 | ~25 min (RTX 6000D) |
| 类别数 | 29 |
| 训练/验证图 | 2,335 / 236 |

快速开始：
```bash
conda create -n ai4agri python=3.10 -y
conda activate ai4agri
pip install ultralytics
cd crop-detect-yolov8
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').train(data='data/plantdoc/data.yaml', epochs=20, imgsz=640, batch=64, device=0)"
```

---

## 🛰️ 阶段 3B：遥感 + GIS 管线 ✅

**仓库子目录**：`remote-sensing-gis/` + `stage3b-report/`

把 Sentinel-2 真实卫星影像加工成**河南省县域植被健康专题图**。

```
┌─────────────────────┐          ┌─────────────────────┐
│    边界管线          │          │    遥感管线          │
│ DataV GeoJSON       │          │ Planetary Computer  │
│   ↓                 │          │   ↓                 │
│ 递归展平 + shapely  │          │ STAC search         │
│   ↓                 │          │   ↓                 │
│ to_crs(UTM50N)      │          │ /vsicurl/ 远程裁剪   │
│   ↓                 │          │   ↓                 │
│ henan_counties.shp  │          │ NDVI = (B08-B04)/.. │
└────────┬────────────┘          └────────┬────────────┘
         └──────── Zonal Statistics ───────┘
                          ↓
                  县域 NDVI 专题图
```

### 核心技术决策

| 选型 | 方案 | 原因 |
|---|---|---|
| Sentinel-2 数据源 | **Planetary Computer**（微软） | 零注册 + 国内直连 + COG 远程裁剪 |
| 行政边界 | **DataV GeoAtlas**（阿里） | 零注册 + 河南 adcode=410000 |
| CRS | **WGS84 → UTM Zone 50N** | 遥感计算必须用米制投影 |
| CRS 构造 | **`CRS.from_proj4()`** | 绕过 pyproj 3.7.x from_epsg bug |
| 远程读取 | **`/vsicurl/` + `from_bounds()`** | 不下载整景 1.2GB |

### 验证结果（周口，2024-05-06 冬小麦季）

| 指标 | 值 |
|---|---|
| 均值 NDVI | 0.311 |
| NDVI 范围 | [-0.014, 0.598] |
| 分辨率 | 10m |
| 茂盛作物区 (NDVI>0.4) | 33.5% |

### 📄 项目报告

[HTML 完整报告](stage3b-report/stage3b-report.html)

---

## 📈 阶段 3C：多指数 + Folium + 时序变化 ✅

**仓库子目录**：`remote-sensing-gis/scripts/` + `remote-sensing-gis/outputs/`

### 多指数扩展（周口，2024-05-06）

| 指数 | 公式 | 均值 | 范围 | 语义 |
|---|---|---|---|---|
| **NDVI** | (NIR-R)/(NIR+R) | 0.311 | [-0.014, 0.598] | 植被活力 |
| **EVI** | 2.5*(NIR-R)/(NIR+6R-7.5B+1) | 0.398 | [-0.020, 1.000] | 增强型（抗土壤） |
| **SAVI** | (NIR-R)/(NIR+R+0.5)×1.5 | 0.260 | [-0.010, 0.536] | 土壤调节（裸土区） |
| **NDWI** | (G-NIR)/(G+NIR) | -0.310 | [-0.548, 0.090] | 水体检测 |

### 时序变化（周口，春 vs 秋）

| 时间点 | 均值 NDVI | 对应物候 | 说明 |
|---|---|---|---|
| **2024-04-18 春季** | **0.419** | 冬小麦灌浆期 | 植被最茂盛 |
| **2024-10-08 秋季** | **0.105** | 秋收后休耕地 | 裸土为主 |
| **差值 (春-秋)** | **+0.314** | 显著正值 | 清晰揭示小麦季 ✅ |

### Folium 交互式地图

`zhoukou_interactive_map.html` — 4 指数图层可切换，县域悬停显示统计值，全屏浏览。

### 脚本清单

```
scripts/
├── download_real_ndvi.py       # 3B 基线验证 (✅)
├── zhoukou_multi_indices.py    # 3C 多指数管线 (✅)
├── make_interactive_map.py     # 3C Folium HTML (✅)
├── zhoukou_timeseries.py       # 3C 春 vs 秋时序 (✅)
├── henan5_multi_indices.py     # 河南中南部 5 市 (🚧 骨架，大窗口慢)
├── henan_full_indices.py       # 全省管线 (🚧 OOM)
└── ...
```

### 产出目录

```
outputs/
├── sentinel2_real/             # 3B 基线 NDVI
├── zhoukou_multi/              # ✅ 4 指数 tif + shp + Folium HTML
├── zhoukou_timeseries/         # ✅ 春/秋 NDVI + 差值 tif + 三联图 + 直方图
├── henan5_multi/               # 🚧 河南中南部 5 市
└── henan_full/                 # 🚧 全省
```

### 快速开始

```bash
pip install geopandas rasterio pyproj shapely pystac-client planetary-computer rasterstats folium matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple

cd remote-sensing-gis
python scripts/zhoukou_multi_indices.py   # ✅ 多指数管线
python scripts/zhoukou_timeseries.py       # ✅ 春 vs 秋时序
python scripts/make_interactive_map.py     # ✅ Folium 交互式地图
```

### 网络注意事项

Planetary Computer（微软 Blob Storage）国内直连不稳定：
- **bbox < 0.2°×0.15°（~2km）**：远程裁剪 ~15-30s/波段，稳定可用 ✅
- **bbox > 1° 或完整 tile（10980×10980）**：超时或 10+ min ❌
- **解决**：严格限制 bbox 在单个 UTM zone 内（±3° 中央经线）

---

## 📂 完整目录结构

```
ai4agri-stage3/
├── README.md                        ← 你正在看
│
├── crop-detect-yolov8/              # ✅ 阶段 3A YOLOv8
│   ├── README.md
│   ├── best.pt                      # 模型权重
│   ├── results-gpu/                 # 训练产出
│   └── data/plantdoc/               # PlantDoc 数据集
│
├── remote-sensing-gis/              # ✅ 阶段 3B+3C 遥感 GIS
│   ├── scripts/                     # 12 个 .py 脚本
│   │   ├── download_real_ndvi.py        # ✅ 基线 NDVI
│   │   ├── zhoukou_multi_indices.py     # ✅ 多指数
│   │   ├── zhoukou_timeseries.py        # ✅ 时序变化
│   │   ├── make_interactive_map.py      # ✅ Folium 地图
│   │   └── ...
│   └── outputs/
│       ├── sentinel2_real/              # 3B 基线产出
│       ├── zhoukou_multi/               # ✅ 4 指数 tif+shp+HTML
│       ├── zhoukou_timeseries/          # ✅ 春/秋/差值 tif + PNG
│       ├── henan5_multi/                # 🚧 骨架
│       └── henan_full/                  # 🚧 骨架
│
└── stage3b-report/                  # 📄 阶段 3B HTML 报告
    └── stage3b-report.html
```

---

## 🛣️ 路线图

```
✅ 阶段 1  Python + ML 回归 (Kaggle Crop Yield)
✅ 阶段 2  PyTorch + CNN 分类 (PlantVillage 91.58%)
✅ 阶段 3A YOLOv8 目标检测 (mAP50 62.05%)
✅ 阶段 3B 遥感 + GIS 管线 (Sentinel-2 NDVI)
✅ 阶段 3C 多指数 + Folium + 时序变化
⬜ 阶段 4  简历 + 项目作品集
```

---

MIT License © 2026 tang2277 · 作物栽培硕士 → AI for Science 工程师
