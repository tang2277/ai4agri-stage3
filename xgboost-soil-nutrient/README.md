# XGBoost 土壤养分管理 🌱

> 用 XGBoost / LightGBM / CatBoost 预测 6 种关键土壤养分 + 作物产量

---

## 项目概述

**1500 样点 × 20 特征 → 6 目标回归** 多输出任务，土壤养分管理的完整机器学习管线。

```
目标变量 (6):
├── SOC_g_kg           — 土壤有机碳 (g/kg)
├── total_N_g_kg       — 全氮 (g/kg)
├── available_P_mg_kg  — 有效磷 (mg/kg)
├── available_K_mg_kg  — 速效钾 (mg/kg)
├── pH                 — 酸碱度
└── yield_kg_ha        — 作物产量 (kg/ha)

特征变量 (20):
├── 地形/气候: elevation, slope, precipitation, temperature, annual_GDD
├── 遥感:      NDVI_spring, NDVI_autumn
├── 土壤物理:  pH_water, clay_pct, silt_pct, sand_pct, bulk_density, CEC, base_saturation
├── 管理:      organic_input, fertilizer_N, fertilizer_P, fertilizer_K
└── 类别:      soil_type_enc, land_use_enc
```

## 快速开始

```bash
conda create -n soil python=3.10 -y
conda activate soil
pip install xgboost lightgbm catboost scikit-learn pandas numpy matplotlib seaborn -i https://pypi.tuna.tsinghua.edu.cn/simple

# 1. 生成模拟数据集 (真实因果关系 + 野外采样缺失值)
python scripts/make_data.py

# 2. 完整训练管线 (EDA → 精调 → 多模型对比 → 可视化)
python scripts/train_xgboost.py
```

## 结果

### XGBoost 精调测试集 R²

| 目标 | R² | RMSE | 单位 | 最优树 |
|---|---|---|---|---|
| **pH** | **0.923** | 0.277 | - | 204 |
| **yield_kg_ha** | **0.880** | 269.1 | kg/ha | 186 |
| **SOC_g_kg** | **0.793** | 0.301 | g/kg | 227 |
| **available_K_mg_kg** | **0.704** | 4.581 | mg/kg | 188 |
| **available_P_mg_kg** | 0.400 | 1.254 | mg/kg | 108 |
| **total_N_g_kg** | 0.392 | 0.094 | g/kg | 74 |

### 多模型对比 (SOC_g_kg)

| 模型 | R² | RMSE | 耗时 |
|---|---|---|---|
| **LightGBM** | **0.785** | 0.307 | 0.50s |
| XGBoost (精调) | 0.780 | 0.311 | 0.80s |
| XGBoost (默认) | 0.774 | 0.315 | 0.66s |
| Random Forest | 0.773 | 0.316 | 0.96s |
| HGB (sklearn) | 0.773 | 0.315 | 0.57s |
| Ridge (线性) | 0.834 | 0.269 | 0.03s |

> 💡 Ridge 在 SOC 上 R²=0.834 反而最好 — 因为 SOC 和 `organic_input` 的关系基本线性。树模型在非线性关系更强的目标（pH, yield）上优势明显。

### XGBoost 土壤数据最优配置

```python
BEST_SOIL_XGB = dict(
    n_estimators=1000,      # 给够，early_stop 自动截断
    learning_rate=0.05,     # 小步多树 = 更好泛化
    max_depth=5,            # 土壤特征少，别太深
    min_child_weight=10,    # 小样本防过拟合
    subsample=0.7,          # 每棵树抽 70% 样点
    colsample_bytree=0.7,   # 每棵树抽 70% 特征
    reg_alpha=0.1,          # L1 正则自动选重要特征
    reg_lambda=5.0,         # L2 正则防权重过大
    early_stopping_rounds=50,
    objective='reg:squarederror',
    n_jobs=-1
)
```

## 产出

```
outputs/
├── correlation_heatmap.png        特征相关性热力图
├── feature_importance_SOC.png     SOC 特征重要性 Top 10
├── feature_importance_heatmap.png 6 目标特征重要性热力图
├── prediction_scatter.png         6 目标 预测 vs 真实 散点图
├── model_comparison.png           多模型 R²/RMSE/MAE 对比
├── model_comparison.csv           多模型对比数据表
├── metrics.csv                     6 目标完整评估指标
└── results.json                    完整结果 JSON
```

## 项目结构

```
xgboost-soil-nutrient/
├── README.md
├── requirements.txt
├── data/
│   └── soil_nutrient.csv          1500 × 26
├── outputs/                       训练产出 (PNG + CSV + JSON)
└── scripts/
    ├── make_data.py               生成模拟土壤数据集
    └── train_xgboost.py           完整训练管线 (EXITCODE=0 ✅)
```

## 面试亮点

1. **XGBoost/LightGBM/CatBoost/RF/HGB/Ridge 6 模型对比** — 展示你不只会一个模型，还能根据数据选对模型
2. **特征重要性分析** — 土壤养分管理最关心"哪些因素影响大"，这个项目直接给答案
3. **缺失值处理** — 1% 缺失率，XGBoost 原生支持 NaN，对比 Ridge 需要手动 dropna
4. **6 目标多输出** — 不是单目标，展示你能处理多任务
5. **调参策略** — 按土壤数据特性定 max_depth=5, min_child_weight=10 等

## 可扩展方向

- [ ] 接入真实土壤普查数据（中国农科院或各省土壤普查办公室公开数据）
- [ ] 加入 Sentinel-2 时序 NDVI 特征（和你阶段 3C 的遥感管线对接！）
- [ ] 地理加权回归 (GWR) vs XGBoost — 土壤属性有空间自相关
- [ ] 施肥推荐系统（多输出 → 决策树推理 N/P/K 推荐量）
- [ ] FastAPI 部署 + 前端交互

---

MIT License © 2026 tang2277
