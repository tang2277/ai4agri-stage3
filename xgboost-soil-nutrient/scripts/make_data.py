"""生成模拟但真实的土壤养分数据集（1500 样点，6 目标回归）"""
import os, numpy as np, pandas as pd
from sklearn.datasets import make_regression

np.random.seed(42)
n = 1500

# 1. 生成基础特征
X_raw, _ = make_regression(n_samples=n, n_features=18, n_informative=12, noise=1.5, random_state=42)

feature_names = [
    'elevation', 'slope', 'precipitation', 'temperature', 'annual_GDD',
    'NDVI_spring', 'NDVI_autumn', 'pH_water', 'clay_pct', 'silt_pct',
    'sand_pct', 'bulk_density', 'CEC', 'base_saturation',
    'organic_input', 'fertilizer_N', 'fertilizer_P', 'fertilizer_K'
]
X = pd.DataFrame(X_raw, columns=feature_names)

# 2. 先给物理意义：clip 到合理范围（模拟真实土壤属性）
X['elevation'] = np.clip(X['elevation'] * 50 + 350, 50, 1500)       # m
X['slope']     = np.clip(np.abs(X['slope']) * 5, 0, 45)             # degree
X['precipitation'] = np.clip(X['precipitation'] * 100 + 800, 200, 2000)  # mm
X['temperature']   = np.clip(X['temperature'] * 3 + 15, -5, 30)      # °C
X['NDVI_spring']   = np.clip(X['NDVI_spring'] * 0.3 + 0.4, 0.05, 0.85)
X['NDVI_autumn']   = np.clip(X['NDVI_autumn'] * 0.3 + 0.2, 0.02, 0.70)
X['pH_water']      = np.clip(X['pH_water'] * 1.5 + 6.5, 4.0, 9.0)
X['clay_pct']      = np.clip(X['clay_pct'] * 15 + 25, 5, 70)
X['silt_pct']      = np.clip(X['silt_pct'] * 15 + 35, 5, 70)
X['sand_pct']      = 100 - X['clay_pct'] - X['silt_pct']              # 保证三组分和=100
X['sand_pct']      = np.clip(X['sand_pct'], 5, 80)
X['bulk_density']  = np.clip(X['bulk_density'] * 0.1 + 1.3, 0.9, 1.8)
X['CEC']           = np.clip(X['CEC'] * 5 + 15, 2, 45)                # cmolc/kg
X['base_saturation'] = np.clip(X['base_saturation'] * 20 + 60, 10, 99)
X['organic_input'] = np.clip(X['organic_input'] * 0.5 + 2.0, 0, 10)
X['fertilizer_N']  = np.clip(X['fertilizer_N'] * 30 + 100, 0, 300)    # kg/ha
X['fertilizer_P']  = np.clip(X['fertilizer_P'] * 15 + 50, 0, 150)
X['fertilizer_K']  = np.clip(X['fertilizer_K'] * 20 + 80, 0, 250)

# 3. 生成目标变量（有因果关系，模拟真实土壤过程）
#    注意：降低噪声 + 增强特征-目标关系，让 XGBoost 能学好
SOC = (0.8 * X['organic_input']
       + 0.6 * X['NDVI_spring'] * 2.5
       + 0.5 * X['CEC'] * 0.08
       - 0.3 * X['temperature'] * 0.08
       + 0.4 * X['clay_pct'] * 0.02
       + np.random.normal(0, 0.25, n))
SOC = np.clip(SOC, 0.2, 8.0)

total_N = (0.12 * SOC
           + 0.05 * X['fertilizer_N'] * 0.01
           + 0.05 * X['NDVI_spring']
           + 0.02 * X['organic_input']
           + np.random.normal(0, 0.08, n))
total_N = np.clip(total_N, 0.05, 0.8)

avail_P = (1.5 * SOC
           + 0.04 * X['fertilizer_P']
           + 0.15 * X['pH_water']
           - 0.8 * X['sand_pct'] * 0.01
           + 0.08 * X['organic_input']
           + np.random.normal(0, 1.2, n))
avail_P = np.clip(avail_P, 0.5, 50)

avail_K = (1.2 * X['CEC']
           + 2.0 * SOC
           + 0.08 * X['fertilizer_K']
           + 0.15 * X['base_saturation']
           + np.random.normal(0, 4.0, n))
avail_K = np.clip(avail_K, 10, 250)

pH_target = (4.5 + 0.05 * X['base_saturation']
             - 0.003 * X['precipitation']
             + 0.002 * X['elevation']
             + np.random.normal(0, 0.25, n))
pH_target = np.clip(pH_target, 4.0, 9.0)

yield_kg = (800 * SOC
            + 1500 * X['NDVI_spring']
            + 1.2 * avail_K
            + 0.5 * X['precipitation']
            - 30 * X['temperature'] * 0.5
            + np.random.normal(0, 150, n))
yield_kg = np.clip(yield_kg, 500, 8000)

# 4. 类别特征
soil_types = np.select([
    X['clay_pct'] > 40,
    X['sand_pct'] > 60,
    X['clay_pct'] < 20
], ['Clay', 'Sandy', 'Silty'], default='Loam')

land_use = np.random.choice(['Cropland', 'Forest', 'Grassland', 'Urban'], n,
                             p=[0.55, 0.25, 0.15, 0.05])

# 5. 组合
targets = pd.DataFrame({
    'SOC_g_kg': np.round(SOC, 3),
    'total_N_g_kg': np.round(total_N, 3),
    'available_P_mg_kg': np.round(avail_P, 2),
    'available_K_mg_kg': np.round(avail_K, 1),
    'pH': np.round(pH_target, 2),
    'yield_kg_ha': np.round(yield_kg, 0),
})

df = pd.concat([targets, X.round(3),
                pd.Series(soil_types, name='soil_type'),
                pd.Series(land_use, name='land_use')], axis=1)

# 6. 加缺失值（模拟野外采样缺测）
rng = np.random.RandomState(42)
for col, rate in [('SOC_g_kg', 0.06), ('available_K_mg_kg', 0.05),
                   ('bulk_density', 0.08), ('CEC', 0.04),
                   ('yield_kg_ha', 0.02)]:
    mask = rng.random(n) < rate
    df.loc[mask, col] = np.nan

# 7. 保存
out_dir = r"C:\Users\Leng\Documents\Trae-CN\default\xgboost-soil-nutrient\data"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'soil_nutrient.csv')
df.to_csv(out_path, index=False)

print(f"✅ 数据集: {out_path}")
print(f"   Shape: {df.shape}")
print(f"   目标 (6): {list(targets.columns)}")
print(f"   特征: {len(X.columns) + 2} 个")
print(f"   缺失值: {df.isna().sum().sum()} ({df.isna().mean().mean()*100:.1f}%)")
print(f"\n目标变量统计:")
print(df[list(targets.columns)].describe().round(3).to_string())
print(f"\n土壤类型分布:")
print(df['soil_type'].value_counts().to_string())
