# AI for Agriculture · Stage 3

> YOLOv8 作物病害检测 + Sentinel-2 遥感 GIS 管线

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

## 🛰️ 阶段 3B：遥感 + GIS 管线 🚧

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

### 快速开始

```bash
pip install geopandas rasterio pyproj shapely pystac-client planetary-computer rasterstats matplotlib pandas -i https://pypi.tuna.tsinghua.edu.cn/simple

cd remote-sensing-gis
python scripts/download_real_ndvi.py   # ✅ 周口 NDVI 已验证
python scripts/henan_county_ndvi.py    # 🚧 全省管线
```

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

## 📂 完整目录结构

```
ai4agri-stage3/
├── README.md                        ← 你正在看
│
├── crop-detect-yolov8/              # ✅ 阶段 3A YOLOv8
│   ├── README.md                    # YOLOv8 详细文档
│   ├── DELIVERY_SUMMARY.md          # 交付总结
│   ├── best.pt                      # 模型权重
│   ├── results-gpu/                 # 训练产出
│   └── data/plantdoc/               # PlantDoc 数据集
│
├── remote-sensing-gis/              # 🚧 阶段 3B 遥感 GIS
│   └── scripts/
│       ├── download_real_ndvi.py    # ✅ 已验证
│       ├── henan_county_ndvi.py      # 🚧 进行中
│       └── stage3b_full_pipeline.py  # 管线骨架
│
└── stage3b-report/                  # 📄 阶段 3B HTML 报告
    └── stage3b-report.html
```

---

## 🛣️ 路线图

```
✅ 阶段 1  Python + ML 回归 (Kaggle Crop Yield)
✅ 阶段 2  PyTorch + CNN 分类 (PlantVillage 91.58%)
✅ 阶段 3A YOLOv8 目标检测 (本项目 mAP50 62.05%)
🚧 阶段 3B 遥感 + GIS (Sentinel-2 NDVI 专题图)
⬜ 阶段 4  简历 + 项目作品集
```

---

MIT License © 2026 tang2277 · 作物栽培硕士 → AI for Science 工程师
