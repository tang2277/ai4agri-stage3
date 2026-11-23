"""
XGBoost 土壤养分管理 · 完整 Pipeline
====================================

6 个目标变量多输出回归 + XGBoost 精调 + 多模型对比 + 可视化
目标: SOC (g/kg), total_N, available_P, available_K, pH, yield (kg/ha)
"""

import os, warnings, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
import time

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ============ 配置 ============
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'soil_nutrient.csv')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

TARGETS = ['SOC_g_kg', 'total_N_g_kg', 'available_P_mg_kg',
           'available_K_mg_kg', 'pH', 'yield_kg_ha']
RANDOM_STATE = 42

print("=" * 70)
print("🌱  XGBoost 土壤养分管理 · 完整 Pipeline")
print("=" * 70)

# ============ 1. 加载 + 探索 ============
print("\n📥 Step 1: 加载数据")
df = pd.read_csv(DATA_PATH)
print(f"   Shape: {df.shape[0]} 样点 × {df.shape[1]} 列")
print(f"   缺失值: {df.isna().sum().sum()} ({df.isna().mean().mean()*100:.1f}%)")

# 类别特征编码
le_soil = LabelEncoder()
le_land = LabelEncoder()
df['soil_type_enc'] = le_soil.fit_transform(df['soil_type'])
df['land_use_enc'] = le_land.fit_transform(df['land_use'])

# ============ 2. 特征工程 ============
print("\n🔧 Step 2: 特征工程")
feature_cols = [c for c in df.columns if c not in TARGETS + ['soil_type', 'land_use']]
print(f"   特征数: {len(feature_cols)}")
print(f"   {feature_cols}")

# 相关性热力图
fig, ax = plt.subplots(figsize=(14, 12))
corr = df[TARGETS + feature_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn_r',
            center=0, square=True, ax=ax, annot_kws={'size': 7})
ax.set_title('特征相关性热力图 (含目标变量)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'correlation_heatmap.png'), dpi=150)
plt.close()
print("   ✅ 相关性热力图已保存")

# ============ 3. 准备数据 ============
print("\n✂️  Step 3: 训练/测试集划分")
X = df[feature_cols].copy()
y = df[TARGETS].copy()

# 每个目标各自有缺失值 — 对每个目标分别 dropna
train_test_sets = {}
for tgt in TARGETS:
    mask = y[tgt].notna()
    X_tgt = X[mask].copy()
    y_tgt = y.loc[mask, tgt].copy()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_tgt, y_tgt, test_size=0.2, random_state=RANDOM_STATE
    )
    train_test_sets[tgt] = (X_tr, X_te, y_tr, y_te)
    print(f"   {tgt}: 训练 {len(X_tr)}, 测试 {len(X_te)}")

# ============ 4. XGBoost 精调 ============
print("\n🚀 Step 4: XGBoost 精调 (每个目标独立)")

# 土壤数据专用最优配置（来自之前调参经验）
XGB_BASE = dict(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    min_child_weight=10,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=5.0,
    early_stopping_rounds=50,
    objective='reg:squarederror',
    eval_metric=['rmse'],
    n_jobs=-1,
    random_state=RANDOM_STATE
)

results = {}
feature_importances_all = {}

for tgt in TARGETS:
    X_tr, X_te, y_tr, y_te = train_test_sets[tgt]
    
    model = xgb.XGBRegressor(**XGB_BASE)
    
    t0 = time.time()
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    train_time = time.time() - t0
    
    # 预测
    y_pred_tr = model.predict(X_tr)
    y_pred_te = model.predict(X_te)
    
    # 评估
    metrics = {
        'R2_train': r2_score(y_tr, y_pred_tr),
        'R2_test':  r2_score(y_te, y_pred_te),
        'RMSE_train': mean_squared_error(y_tr, y_pred_tr)**0.5,
        'RMSE_test':  mean_squared_error(y_te, y_pred_te)**0.5,
        'MAE_train':  mean_absolute_error(y_tr, y_pred_tr),
        'MAE_test':   mean_absolute_error(y_te, y_pred_te),
        'best_iteration': model.best_iteration,
        'train_time_s': round(train_time, 2),
    }
    results[tgt] = metrics
    feature_importances_all[tgt] = dict(zip(feature_cols, model.feature_importances_))
    
    # 打印
    unit = {'SOC_g_kg': 'g/kg', 'total_N_g_kg': 'g/kg',
            'available_P_mg_kg': 'mg/kg', 'available_K_mg_kg': 'mg/kg',
            'pH': '', 'yield_kg_ha': 'kg/ha'}[tgt]
    print(f"\n   📌 {tgt} ({unit})")
    print(f"      R²  训练={metrics['R2_train']:.3f}  测试={metrics['R2_test']:.3f}")
    print(f"      RMSE 训练={metrics['RMSE_train']:.3f}  测试={metrics['RMSE_test']:.3f} {unit}")
    print(f"      MAE  训练={metrics['MAE_train']:.3f}  测试={metrics['MAE_test']:.3f} {unit}")
    print(f"      最优树: {metrics['best_iteration']}  耗时: {metrics['train_time_s']}s")

# ============ 5. 多模型对比 ============
print("\n⚔️  Step 5: 多模型对比 (1 target = SOC_g_kg)")

tgt = 'SOC_g_kg'
X_tr, X_te, y_tr, y_te = train_test_sets[tgt]

# Ridge 不接受 NaN — 先 dropna
mask_tr = X_tr.notna().all(axis=1)
mask_te = X_te.notna().all(axis=1)
X_tr_clean, y_tr_clean = X_tr[mask_tr], y_tr[mask_tr]
X_te_clean, y_te_clean = X_te[mask_te], y_te[mask_te]

models = {
    'XGBoost (精调)': xgb.XGBRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=5,
        min_child_weight=10, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=5.0,
        n_jobs=-1, random_state=RANDOM_STATE, objective='reg:squarederror'),
    'XGBoost (默认)': xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.1, max_depth=6,
        n_jobs=-1, random_state=RANDOM_STATE, objective='reg:squarederror'),
    'LightGBM':       lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                                         max_depth=5, random_state=RANDOM_STATE, verbose=-1),
    'Random Forest':  RandomForestRegressor(n_estimators=500, max_depth=10,
                                             random_state=RANDOM_STATE, n_jobs=-1),
    'HGB (sklearn)': HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05,
                                                    max_depth=5, random_state=RANDOM_STATE),
    'Ridge (线性)':   Ridge(alpha=1.0),
}

compare_df = []
for name, model in models.items():
    t0 = time.time()
    # 树模型可接受 NaN，Ridge 用 clean 数据
    if 'Ridge' in name:
        model.fit(X_tr_clean, y_tr_clean)
        y_pred = model.predict(X_te_clean)
        y_true = y_te_clean
    else:
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_true = y_te
    fit_time = time.time() - t0
    row = {
        '模型': name,
        'R²': r2_score(y_true, y_pred),
        'RMSE': mean_squared_error(y_true, y_pred)**0.5,
        'MAE': mean_absolute_error(y_true, y_pred),
        '训练耗时(s)': round(fit_time, 2),
    }
    compare_df.append(row)
    print(f"   {name:<20} R²={row['R²']:.3f}  RMSE={row['RMSE']:.3f}  耗时={row['训练耗时(s)']}s")

compare_df = pd.DataFrame(compare_df)
compare_df.to_csv(os.path.join(OUT_DIR, 'model_comparison.csv'), index=False)

# ============ 6. 可视化 ============
print("\n🎨 Step 6: 生成可视化")

# 6.1 特征重要性 Top 10（SOC）
fig, ax = plt.subplots(figsize=(10, 6))
fi_soc = pd.Series(feature_importances_all['SOC_g_kg']).sort_values(ascending=True)
fi_soc.tail(10).plot.barh(ax=ax, color='#2d6a4f')
ax.set_xlabel('重要性')
ax.set_title('XGBoost 特征重要性 Top 10 — SOC 预测', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'feature_importance_SOC.png'), dpi=150)
plt.close()
print("   ✅ SOC 特征重要性")

# 6.2 特征重要性汇总热力图
fi_df = pd.DataFrame(feature_importances_all)
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(fi_df, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
            annot_kws={'size': 8})
ax.set_title('特征重要性热力图 — 6 个目标变量', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'feature_importance_heatmap.png'), dpi=150)
plt.close()
print("   ✅ 特征重要性热力图")

# 6.3 预测 vs 真实散点图（6 目标）
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
unit_map = {'SOC_g_kg': 'g/kg', 'total_N_g_kg': 'g/kg', 'available_P_mg_kg': 'mg/kg',
            'available_K_mg_kg': 'mg/kg', 'pH': '', 'yield_kg_ha': 'kg/ha'}
for idx, tgt in enumerate(TARGETS):
    ax = axes[idx // 3][idx % 3]
    X_tr, X_te, y_tr, y_te = train_test_sets[tgt]
    # 用已训练的模型（重新 fit 一次快速）
    m = xgb.XGBRegressor(**XGB_BASE)
    m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    y_pred = m.predict(X_te)
    
    ax.scatter(y_te, y_pred, alpha=0.5, s=20, color='#2d6a4f', edgecolors='none')
    # y=x 线
    lim_min = min(y_te.min(), y_pred.min())
    lim_max = max(y_te.max(), y_pred.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', lw=1.5, label='理想预测线')
    ax.set_xlabel(f'真实值 ({unit_map[tgt]})')
    ax.set_ylabel(f'预测值 ({unit_map[tgt]})')
    r2 = r2_score(y_te, y_pred)
    ax.set_title(f'{tgt}\nR² = {r2:.3f}', fontsize=11)
    ax.legend(fontsize=9)
plt.suptitle('XGBoost 预测 vs 真实值（测试集）', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'prediction_scatter.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ 预测散点图")

# 6.4 多模型对比柱状图
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, metric in enumerate(['R²', 'RMSE', 'MAE']):
    ax = axes[idx]
    bar_colors = ['#2d6a4f' if v == compare_df[metric].max() and metric == 'R²' 
                  else '#e76f51' if v == compare_df[metric].min() and metric != 'R²'
                  else '#52b788' for v in compare_df[metric]]
    bars = ax.bar(compare_df['模型'], compare_df[metric], color=bar_colors)
    ax.set_title(f'{metric} 对比（SOC 测试集）', fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', rotation=35, labelsize=8)
    for bar, val in zip(bars, compare_df[metric]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'model_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ 多模型对比柱状图")

# ============ 7. 保存结果 ============
print("\n💾 Step 7: 保存结果")

# 指标表
results_df = pd.DataFrame(results).T
results_df.index.name = 'target'
results_df.to_csv(os.path.join(OUT_DIR, 'metrics.csv'))
print("   ✅ metrics.csv")

# JSON 详情
with open(os.path.join(OUT_DIR, 'results.json'), 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("   ✅ results.json")

print("\n" + "=" * 70)
print("🎉  完成！产出在 outputs/ 目录下")
print("=" * 70)
print("\n📊 XGBoost 精调结果汇总:")
print(f"{'目标':<22} {'R²(test)':>10} {'RMSE':>10} {'最优树':>8}")
print("-" * 52)
for tgt in TARGETS:
    r = results[tgt]
    print(f"{tgt:<22} {r['R2_test']:>10.3f} {r['RMSE_test']:>10.3f} {r['best_iteration']:>8}")
