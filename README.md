# AI for Agriculture · Stage 3

> 遥感 + GIS 管线：Sentinel-2 真实影像 → NDVI → 河南省县域专题图

## 📂 目录结构

```
ai4agri-stage3/
├── remote-sensing-gis/          # 🧪 阶段 3B 核心管线
│   └── scripts/                 # Python 脚本
│       ├── download_real_ndvi.py    # ✅ 已验证：周口 NDVI 裁剪
│       ├── henan_county_ndvi.py     # 🚧 进行中：全省管线
│       └── stage3b_full_pipeline.py # 管线骨架
│
├── stage3b-report/              # 📄 项目报告（HTML）
│   └── stage3b-report.html
│
└── README.md                    # 本文件
```

## 🎯 阶段 3B 目标

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
                          │
                          ↓
                  县域 NDVI 专题图
```

## 🔑 核心技术决策

| 选型 | 方案 | 原因 |
|---|---|---|
| Sentinel-2 数据源 | **Planetary Computer**（微软） | 零注册 + 国内直连 + COG 远程裁剪 |
| 行政边界 | **DataV GeoAtlas**（阿里） | 零注册 + 河南 adcode=410000 |
| CRS | **WGS84 → UTM Zone 50N** | 遥感计算必须用米制投影 |
| CRS 构造 | **`CRS.from_proj4()`** | 绕过 pyproj 3.7.x from_epsg bug |
| 远程读取 | **`/vsicurl/` + `from_bounds()`** | 不下载整景 1.2GB |
| Zonal Stats | **rasterstats** | 开源分区统计 |

## 🚀 快速开始

```bash
conda create -n ai4agri python=3.10 -y
conda activate ai4agri
pip install geopandas rasterio pyproj shapely pystac-client planetary-computer rasterstats matplotlib pandas -i https://pypi.tuna.tsinghua.edu.cn/simple

cd remote-sensing-gis
python scripts/download_real_ndvi.py   # ✅ 周口 NDVI（已验证）
python scripts/henan_county_ndvi.py    # 🚧 全省管线
```

## 📊 验证结果（周口，2024-05-06 冬小麦季）

| 指标 | 值 |
|---|---|
| 均值 NDVI | 0.311 |
| NDVI 范围 | [-0.014, 0.598] |
| 分辨率 | 10m |
| 茂盛作物区 (NDVI>0.4) | 33.5% |

## 📚 项目报告

见 [stage3b-report.html](stage3b-report/stage3b-report.html)。

*AI for Agriculture · Stage 3B · 2026*
