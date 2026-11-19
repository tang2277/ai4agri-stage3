# AI for Science · 阶段 3A 交付总结

> **目标检测入门** | YOLOv8-Nano | PlantDoc 29 类作物病害检测
> 交付日期：2026-11-19 | 作者：tang2277（作物栽培硕士 → AI 工程师转岗）

---

## 一、项目定位

阶段 3A 是 **农业 AI 工程师入门路线** 的进阶实战环节，承接阶段 2（图像分类），首次完整跑通 **YOLOv8 目标检测 + GPU 云训练 + 带框标注数据集** 全流程，从"识别是什么"跨越到"定位在哪里 + 识别是什么"。

| 维度 | 阶段 2 | 阶段 3A（本项目） |
|------|--------|-------------------|
| 方向 | 图像分类（图像级标签） | 目标检测（框 + 类别） |
| 框架 | PyTorch + torchvision | Ultralytics YOLOv8 |
| 数据集 | PlantVillage（54,306 张分类图） | PlantDoc（2,571 张检测图） |
| 模型 | MobileNetV3-Small | YOLOv8-Nano |
| 训练设备 | CPU（~50 min） | AutoDL RTX 6000D GPU |
| 核心指标 | 验证准确率 = 91.58% | mAP50 = **62.05%** |

---

## 二、核心交付物清单

### 2.1 文件级交付明细

```
crop-detect-yolov8/
├── DELIVERY_SUMMARY.md            ← 本交付总结
├── README.md                       ← 项目说明文档
├── data/plantdoc/                  ← 本地 PlantDoc 数据集
│   ├── data.yaml                   ← 数据集配置（本地路径版）
│   ├── images/train/               ← 2,335 张训练图
│   ├── images/val/                 ← 236 张验证图
│   ├── labels/train/               ← YOLO 格式标注
│   └── labels/val/
├── plantdoc-for-gpu/data.yaml      ← GPU 云实例版数据配置
├── results-gpu/                    ← GPU 训练完整产出
│   ├── args.yaml                   ← 训练参数快照
│   ├── results.csv                 ← 20 epoch 逐项日志
│   ├── results.png                 ← Loss + mAP 训练曲线
│   ├── confusion_matrix.png        ← 29 类混淆矩阵
│   ├── BoxF1_curve.png             ← F1-Confidence 曲线
│   └── weights/
│       ├── best.pt                 ← mAP50 最佳权重（~6 MB）
│       └── last.pt                 ← 第 20 epoch 权重
├── predictions-gpu/                ← GPU 模型验证集预测可视化
│   ├── gpu_val_00000.jpg
│   ├── gpu_val_00001.jpg
│   ├── gpu_val_00002.jpg
│   └── gpu_val_00003.jpg
└── predictions/                    ← 阶段过渡性预测产出
    ├── coco-pretrained/            ← COCO 预训练权重测试
    ├── coco8-demo/                 ← YOLOv8 COCO demo
    └── pred_3ep_val_*.jpg          ← CPU 3 epoch 快速测试
```

---

## 三、技术方案与配置

### 3.1 YOLOv8-Nano 模型架构

YOLOv8 是 Ultralytics 2023 年发布的单阶段检测算法，核心改进：

| 模块 | 说明 |
|------|------|
| **Backbone** | Conv + C2f 模块（替代 YOLOv5 C3，梯度流通更高效） |
| **Neck** | PAN-FPN 多尺度特征融合（P3/P4/P5 三层检测头） |
| **Head** | Decoupled Head（分类/回归分支解耦）+ TaskAlignedAssigner 标签分配 |
| **损失函数** | Box Loss（CIoU + DFL）+ Cls Loss（BCE） |

```
YOLOv8-Nano 参数量: ~3.2M
输入尺寸: 640 × 640
检测头: 3 尺度 (80×80, 40×40, 20×20)
```

### 3.2 训练配置

| 配置项 | 值 | 说明 |
|--------|-----|------|
| 预训练权重 | `yolov8n.pt` | COCO 预训练 |
| Epochs | 20 | 快速验证收敛性 |
| Batch size | 64 | RTX 6000D 显存充足 |
| 输入尺寸 | 640 | YOLOv8 默认 |
| 优化器 | auto（SGD） | YOLOv8 默认选择 |
| 初始学习率 | 0.01 | — |
| 最终学习率 | 0.0001 | lrf = 0.01 |
| 预热轮次 | 3 | warmup_epochs |
| 数据增强 | mosaic=1.0, fliplr=0.5, hsv 等 | YOLOv8 默认增强 |
| 训练设备 | AutoDL RTX 6000D (GPU 0) | — |
| AMP | ✅ | 自动混合精度 |
| cls_remap | ✅ | 类别重映射 |

### 3.3 PlantDoc 数据集

| 属性 | 值 |
|------|-----|
| 总图像数 | 2,571 |
| 训练集 | 2,335 张 |
| 验证集 | 236 张 |
| 类别数 | 29 |
| 作物覆盖 | 番茄、玉米、苹果、葡萄、马铃薯、草莓、蓝莓、樱桃、桃子、覆盆子、大豆、南瓜、甜椒 |
| 标注格式 | YOLO 格式（`class x_center y_center width height` 归一化坐标） |

---

## 四、核心结果指标

### 4.1 总体指标（第 20 epoch）

```
🎯 mAP50:        62.05%   (IoU=0.5 条件下的平均精度)
📊 mAP50-95:     48.63%   (IoU=0.5~0.95 区间的平均精度，更严格)
🔍 Precision:    55.81%   (预测为正的里面有多少真的是正的)
📈 Recall:        61.83%   (所有正样本里有多少被成功检测到)
⏱️ 训练总耗时:    ~74 秒 × 20 epochs ≈ 24.7 分钟 (GPU)
📦 模型体积:      ~6 MB (best.pt)
🏆 最佳 epoch:    20 (训练持续收敛)
```

### 4.2 训练损失收敛趋势

| Epoch | Box Loss | Cls Loss | DFL Loss | mAP50 | mAP50-95 |
|-------|----------|----------|----------|-------|----------|
| 1 | 1.470 | 4.707 | 1.669 | 7.78% | 5.62% |
| 5 | 1.240 | 2.028 | 1.354 | 37.54% | 27.38% |
| 10 | 1.163 | 1.568 | 1.301 | 53.07% | 40.09% |
| 15 | 1.080 | 1.197 | 1.234 | 60.08% | 46.96% |
| 20 | **1.029** | **1.000** | **1.207** | **62.05%** | **48.63%** |

> 三类损失均稳步下降，无明显震荡，说明训练稳定收敛。mAP50 从 7.78% → 62.05%，提升约 8 倍。

### 4.3 验证集混淆矩阵洞察

训练产出的 `confusion_matrix.png` 揭示了 29 类间的混淆模式：

| 洞察 | 作物学解读 |
|------|-----------|
| **同作物病害互混率最高** | Tomato leaf ↔ Tomato mold leaf、Potato leaf ↔ Potato leaf early blight — 同一作物不同病害的叶片纹理相似 |
| **健康叶 vs 病叶偶尔混淆** | 部分病斑较轻的样本被预测为健康，与阶段 2 发现一致 |
| **单作物类别识别更精准** | Cherry leaf、Blueberry leaf 等单一病害类别识别率较高，因为叶片形态特征鲜明 |
| **训练 epoch 有限导致边界模糊** | 20 epoch 尚未完全收敛，增加训练轮次有望进一步降低混淆率 |

---

## 五、方法学亮点（面试差异化点）

### 5.1 技术亮点

| # | 亮点 | 价值 |
|---|------|------|
| 1 | **CPU 快速验证 + GPU 正式训练两阶段策略** | 先用 3 epoch 在 CPU 跑通流程（~20 min），确认脚本无误后上 GPU 正式训练，避免 GPU 资源浪费 |
| 2 | **YOLOv8 一站式训练接口** | `model.train()` 单方法封装数据加载、增强、训练、验证、模型保存全流程，降低入门门槛 |
| 3 | **AutoDL GPU 按需计费** | RTX 6000D 按小时计费，训练完成立即关机释放（~25 min 训练仅花费 ≈ ¥1-2） |
| 4 | **YOLOv8 默认数据增强** | Mosaic(1.0) + MixUp(0.0) + Fliplr(0.5) + HSV 扰动，单阶段检测任务的数据增强策略 |
| 5 | **CIoU + DFL 双损失** | YOLOv8 用 Complete IoU Loss + Distribution Focal Loss 进行边界框回归，比传统 IoU 更精准 |

### 5.2 跨阶段对比亮点（分类 vs 检测）

| 维度 | 阶段 2 MobileNetV3 分类 | 阶段 3 YOLOv8 检测 |
|------|------------------------|-------------------|
| 任务粒度 | 图像级（整图一个标签） | 目标级（每个框一个标签） |
| 损失函数 | CrossEntropyLoss | BCE(分类) + CIoU+DFL(回归) |
| 输出形式 | 38 类 Softmax | 3 尺度 × (80²+40²+20²) × (nc+4) |
| 作物学意义 | "这是一张什么病的图" | "图里哪里有什么病" |
| 参数规模 | 2.5M | 3.2M（Nano） |
| mAP 参考 | 不适用 | 62.05% |

### 5.3 作物学视角亮点

| # | 洞察 | 作物学依据 |
|---|------|-----------|
| 1 | **PlantDoc 包含病害 + 健康叶** | 与 PlantVillage（全病害分类）不同，PlantDoc 有健康叶类别，适合做田间巡检工具 |
| 2 | **单叶片拍摄为主** | PlantDoc 多为实验室单叶特写，背景简单，为迈向田间复杂场景检测打好基础 |
| 3 | **29 类覆盖 14 种作物** | 比 PlantVillage（38 类 14 种）少一些病害细分，但多了目标定位能力 |

---

## 六、踩坑记录与解决方案

| # | 问题 | 根因 | 解决方案 |
|---|------|------|----------|
| 1 | **kaggleHub 下载 PlantDoc 速度仅十几 KB/s** | 国内网络 + PyPI 清华源限速 + 海外 Kaggle 中转 | 改为本地打包数据集后直接上传 AutoDL 云实例，上传速度 5-10 MB/s |
| 2 | **data.yaml 路径不匹配** | Windows 本地路径 `C:\...` 与 Linux 云实例 `/root/autodl-tmp/...` 完全不同 | 手动修改 data.yaml 中 path 字段，适配云实例实际路径 |
| 3 | **GitHub 下载 yolov8n.pt 被墙** | GFW 阻断 github.com 直连 | 使用 ghfast.top 镜像前缀下载：`wget https://ghfast.top/https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt` |
| 4 | **GPU 实例关机后担心数据丢失** | AutoDL 关机释放 GPU，但数据盘默认保留 | 数据盘默认持久化存储，关机重开无卡模式即可访问下载 |
| 5 | **多行命令粘贴到 JupyterLab 终端报错** | 终端对多行输入的解析问题 | 改为分步逐条执行，避免一次性粘贴过长命令 |
| 6 | **YOLOv8 detect vs segment 任务混淆** | 默认 task=detect，但数据集配置可能不匹配 | 明确设置 `task: detect`，验证 data.yaml 中 labels 目录存在 |
| 7 | **后台进程持续占用带宽** | kagglehub 启动了后台下载进程无法感知 | 直接关闭终端标签页 + 新建全新终端，比手动 kill 进程更快捷 |

---

## 七、项目运行说明

### 7.1 环境准备

```powershell
# === Conda 环境 ===
conda create -n ai4agri python=3.10 -y
conda activate ai4agri

# === 核心依赖 ===
pip install ultralytics
# ultralytics 会自动安装: torch, torchvision, opencv-python, pyyaml 等

# === GPU 云平台（AutoDL）额外准备 ===
pip install kagglehub    # 数据集下载（可选，推荐本地打包上传）
```

### 7.2 数据集准备

```powershell
# 方式 1：本地已有数据（推荐）
# 确保目录结构为：
# data/plantdoc/
# ├── data.yaml
# ├── images/
# │   ├── train/    (2,335 张)
# │   └── val/      (236 张)
# └── labels/
#     ├── train/    (与 images/train 同名 .txt)
#     └── val/      (与 images/val 同名 .txt)

# 方式 2：KaggleHub 下载（国内慢，不推荐）
# pip install kagglehub
# kagglehub.dataset_download("emmarex/plantdoc")
```

### 7.3 训练命令

```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO("yolov8n.pt")

# 训练（本地 CPU 快速测试）
results = model.train(
    data="data/plantdoc/data.yaml",
    epochs=3,           # CPU 快速验证用 3 epoch
    imgsz=640,
    batch=16,
    device="cpu",
)

# 训练（GPU 正式训练）
results = model.train(
    data="plantdoc-for-gpu/data.yaml",
    epochs=20,
    imgsz=640,
    batch=64,
    device=0,           # GPU 编号
    project="runs",
    name="plantdoc-gpu",
)
```

### 7.4 验证与预测

```python
# 验证最佳模型
model = YOLO("results-gpu/weights/best.pt")
metrics = model.val(data="data/plantdoc/data.yaml")

# 对验证集批量预测并保存
results = model.predict(
    source="data/plantdoc/images/val",
    save=True,
    project="predictions-gpu",
    conf=0.25,
)
```

### 7.5 单图推理

```python
# 单张图片检测
results = model.predict(
    source="path/to/image.jpg",
    save=True,
    conf=0.25,          # 置信度阈值
    iou=0.45,           # NMS IoU 阈值
)

# results[0].boxes 包含所有检测框
# results[0].boxes.xyxy  坐标 (x1,y1,x2,y2)
# results[0].boxes.cls   类别索引
# results[0].boxes.conf 置信度
```

---

## 八、后续路线图

| 阶段 | 内容 | 预计产出 |
|------|------|----------|
| ✅ 阶段 1 | Python + ML 基础（Kaggle Crop Yield） | 回归模型 RMSE=0.4930 |
| ✅ 阶段 2 | PyTorch + CNN + 迁移学习（PlantVillage） | 病害识别 91.58% |
| ✅ **阶段 3A** | **YOLOv8 目标检测 + GPU 云训练** | **PlantDoc mAP50 = 62.05%** |
| ⬜ 阶段 3B | YOLOv8 微调（更多 epoch + 超参搜索） | mAP50 → 70%+ |
| ⬜ 阶段 3C | 遥感 + GIS | Sentinel-2 影像作物分类 |
| ⬜ 阶段 3D | 田间实拍数据集 + 域适应 | 真实场景病害检测 |
| ⬜ 阶段 4 | 简历 + 项目打包 + GitHub Pages | 面试作品集 |

---

## 九、交付验收清单

| # | 验收项 | 状态 | 证据 |
|---|--------|------|------|
| 1 | PlantDoc 数据集本地可用 | ✅ | `data/plantdoc/` 完整目录（2,571 张图 + YOLO 标注） |
| 2 | YOLOv8-Nano 预训练权重 | ✅ | `yolov8n.pt`（COCO 预训练） |
| 3 | GPU 云实例 20 epoch 训练完成 | ✅ | `results-gpu/results.csv` 20 行完整日志 |
| 4 | 最佳模型权重 | ✅ | `results-gpu/weights/best.pt`（~6 MB） |
| 5 | 最终模型权重 | ✅ | `results-gpu/weights/last.pt` |
| 6 | 训练参数快照 | ✅ | `results-gpu/args.yaml`（115 行完整配置） |
| 7 | 训练 Loss/mAP 曲线 | ✅ | `results-gpu/results.png` |
| 8 | 29 类混淆矩阵 | ✅ | `results-gpu/confusion_matrix.png` |
| 9 | F1-Confidence 曲线 | ✅ | `results-gpu/BoxF1_curve.png` |
| 10 | 验证集预测可视化 | ✅ | `predictions-gpu/` 4 张带框标注图 |
| 11 | mAP50 指标 ≥ 60% | ✅ | **62.05%**（超过阶段目标） |
| 12 | 训练数据完整可复现 | ✅ | args.yaml + results.csv + data.yaml 三件套 |
| 13 | 交付总结文档 | ✅ | 本 DELIVERY_SUMMARY.md |
| 14 | 项目 README | ✅ | README.md |

---

<p align="center">
  <sub>Made with 🌾 by tang2277 · 作物栽培硕士 → AI for Science 工程师</sub><br>
  <sub>MIT License · 2026-11-19</sub>
</p>
