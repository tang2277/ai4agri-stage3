# Crop Detect YOLOv8 🌾

> **YOLOv8-Nano 目标检测** | PlantDoc 29 类作物病害定位与识别 | GPU 云训练

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/ultralytics-8.x-green.svg)](https://github.com/ultralytics/ultralytics)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Nano-orange.svg)](https://docs.ultralytics.com/models/yolov8/)
[![mAP50](https://img.shields.io/badge/mAP50-62.05%25-brightgreen.svg)](#核心结果)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](#license)

---

## 📌 项目简介

本项目使用 **YOLOv8-Nano** 目标检测模型，在 **PlantDoc** 数据集上训练 29 类作物病害检测器。训练采用两阶段策略：先在 CPU 上快速验证流程（3 epoch），再在 AutoDL GPU 云平台完成 20 epoch 正式训练，最终 **mAP50 达到 62.05%**。

### 与阶段 2 的区别

| | 阶段 2 MobileNetV3 | **阶段 3 YOLOv8** |
|---|---|---|
| 任务 | 图像分类（整张图一个标签） | **目标检测（定位 + 分类）** |
| 输入 | 分类图像 | **带边界框标注的检测图像** |
| 输出 | 38 类概率 | **N 个检测框（坐标 + 类别 + 置信度）** |
| 指标 | 准确率 91.58% | **mAP50 = 62.05%** |
| 数据 | 54,306 张 | **2,571 张（含 YOLO 标注）** |

---

## 🏆 核心结果

| 指标 | 值 |
|------|-----|
| **mAP50** | **62.05%** |
| mAP50-95 | 48.63% |
| Precision | 55.81% |
| Recall | 61.83% |
| 训练 epoch | 20 |
| 训练耗时 | ~25 min (RTX 6000D) |
| 模型体积 | ~6 MB |
| 类别数 | 29 |
| 训练/验证图 | 2,335 / 236 |

### 训练收敛曲线

> 训练过程中三类损失稳步下降，mAP50 从 7.78% 提升至 62.05%。

| Epoch | Box Loss | Cls Loss | DFL Loss | mAP50 |
|-------|----------|----------|----------|-------|
| 1 | 1.470 | 4.707 | 1.669 | 7.78% |
| 5 | 1.240 | 2.028 | 1.354 | 37.54% |
| 10 | 1.163 | 1.568 | 1.301 | 53.07% |
| 15 | 1.080 | 1.197 | 1.234 | 60.08% |
| **20** | **1.029** | **1.000** | **1.207** | **62.05%** |

---

## 📂 目录结构

```
crop-detect-yolov8/
├── README.md                       ← 你正在看
├── DELIVERY_SUMMARY.md             ← 完整交付总结（面试用）
├── data/plantdoc/                  ← PlantDoc 数据集
│   ├── data.yaml                   ← 本地版配置
│   ├── images/{train,val}/         ← 训练/验证图像
│   └── labels/{train,val}/         ← YOLO 格式标注
├── plantdoc-for-gpu/data.yaml      ← GPU 云实例版配置
├── results-gpu/                    ← GPU 训练产出
│   ├── args.yaml                   ← 训练参数快照
│   ├── results.csv                 ← 20 epoch 日志
│   ├── results.png                 ← Loss/mAP 曲线
│   ├── confusion_matrix.png        ← 29 类混淆矩阵
│   ├── BoxF1_curve.png             ← F1-Confidence 曲线
│   └── weights/{best.pt,last.pt}   ← 模型权重
├── predictions-gpu/                ← 验证集预测可视化
└── predictions/                    ← 过渡性预测产出
```

---

## 🚀 快速开始

### 环境安装

```bash
# 创建 conda 环境
conda create -n ai4agri python=3.10 -y
conda activate ai4agri

# 安装 ultralytics（含 PyTorch）
pip install ultralytics
```

### 数据集准备

PlantDoc 数据集需要以下目录结构（YOLO 格式）：

```
plantdoc/
├── data.yaml          # 指向 train/val 路径 + 29 类名
├── images/
│   ├── train/         # 2,335 张 .jpg
│   └── val/           # 236 张 .jpg
└── labels/
    ├── train/         # 与 images/train 同名 .txt
    └── val/           # 与 images/val 同名 .txt
```

> 下载方式：本地打包上传 AutoDL 最快（5-10 MB/s），避免 kagglehub 国内低速下载。

### 训练

```python
from ultralytics import YOLO

# 加载 COCO 预训练权重
model = YOLO("yolov8n.pt")

# CPU 快速验证（3 epoch，~20 min）
model.train(
    data="data/plantdoc/data.yaml",
    epochs=3, imgsz=640, batch=16, device="cpu"
)

# GPU 正式训练（20 epoch，~25 min on RTX 6000D）
model.train(
    data="plantdoc-for-gpu/data.yaml",
    epochs=20, imgsz=640, batch=64, device=0
)
```

### 验证模型

```python
model = YOLO("results-gpu/weights/best.pt")
metrics = model.val(data="data/plantdoc/data.yaml")
# 输出: mAP50, mAP50-95, Precision, Recall
```

### 预测

```python
# 批量预测
model.predict(
    source="data/plantdoc/images/val",
    save=True, project="predictions-gpu", conf=0.25
)

# 单图检测
results = model.predict(source="test.jpg", conf=0.25)
boxes = results[0].boxes
print(f"检测到 {len(boxes)} 个目标")
print(f"类别: {boxes.cls}")
print(f"置信度: {boxes.conf}")
```

---

## 🔧 训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 模型 | yolov8n.pt | YOLOv8-Nano（~3.2M 参数） |
| Epochs | 20 | GPU 正式训练 |
| Batch size | 64 | RTX 6000D 显存充足 |
| 输入尺寸 | 640×640 | YOLOv8 默认 |
| 优化器 | SGD (auto) | YOLOv8 默认选择 |
| 初始学习率 | 0.01 | — |
| 数据增强 | mosaic, fliplr, hsv | YOLOv8 默认 |
| AMP | ✅ | 自动混合精度加速 |

完整参数见 [`results-gpu/args.yaml`](results-gpu/args.yaml)。

---

## 🗂️ PlantDoc 数据集 29 类

| # | 类别 | 作物 |
|---|------|------|
| 0 | Cherry leaf | 樱桃 |
| 1 | Peach leaf | 桃 |
| 2 | Corn leaf blight | 玉米 |
| 3 | Apple rust leaf | 苹果 |
| 4 | Potato leaf late blight | 马铃薯 |
| 5 | Strawberry leaf | 草莓 |
| 6 | Corn rust leaf | 玉米 |
| 7 | Tomato leaf late blight | 番茄 |
| 8 | Tomato mold leaf | 番茄 |
| 9 | Potato leaf early blight | 马铃薯 |
| 10 | Apple leaf | 苹果 |
| 11 | Tomato leaf yellow virus | 番茄 |
| 12 | Blueberry leaf | 蓝莓 |
| 13 | Tomato leaf mosaic virus | 番茄 |
| 14 | Raspberry leaf | 覆盆子 |
| 15 | Tomato leaf bacterial spot | 番茄 |
| 16 | Squash Powdery mildew leaf | 南瓜 |
| 17 | grape leaf | 葡萄 |
| 18 | Corn Gray leaf spot | 玉米 |
| 19 | Tomato Early blight leaf | 番茄 |
| 20 | Apple Scab Leaf | 苹果 |
| 21 | Tomato Septoria leaf spot | 番茄 |
| 22 | Tomato leaf | 番茄 |
| 23 | Soyabean leaf | 大豆 |
| 24 | Bell_pepper leaf spot | 甜椒 |
| 25 | Bell_pepper leaf | 甜椒 |
| 26 | grape leaf black rot | 葡萄 |
| 27 | Potato leaf | 马铃薯 |
| 28 | Tomato two spotted spider mites leaf | 番茄 |

---

## 💡 技术亮点

| # | 亮点 | 价值 |
|---|------|------|
| 1 | **CPU 验证 + GPU 训练两阶段** | 避免 GPU 资源浪费 |
| 2 | **AutoDL GPU 按需计费** | 25 min 训练 ≈ ¥1-2 |
| 3 | **YOLOv8 C2f 模块** | 替代 YOLOv5 C3，梯度流通更高效 |
| 4 | **CIoU + DFL 边界回归** | 比传统 IoU 更精准 |
| 5 | **一站式 `model.train()`** | 数据加载→增强→训练→验证→保存全封装 |

### 踩坑速查

| 问题 | 解决 |
|------|------|
| kaggleHub 国内下载慢 | 本地打包后上传云实例 |
| data.yaml 路径不匹配 | 手动修改 path 字段适配云实例 |
| GitHub 权重下载被墙 | 使用 ghfast.top 镜像 |
| 关机后数据丢失？ | AutoDL 数据盘默认持久化 |

---

## 🛣️ 路线图

```
✅ 阶段 1  Python + ML 回归 (Kaggle Crop Yield)
✅ 阶段 2  PyTorch + CNN 分类 (PlantVillage 91.58%)
✅ 阶段 3A YOLOv8 目标检测 (本项目 mAP50 62.05%)
⬜ 阶段 3B YOLOv8 微调 → mAP50 70%+
⬜ 阶段 3C 遥感 + GIS (Sentinel-2)
⬜ 阶段 4  简历 + 项目作品集
```

---

## 📄 License

MIT License © 2026 tang2277

---

<p align="center">
  <sub>Made with 🌾 by tang2277 · 作物栽培硕士 → AI for Science 工程师</sub>
</p>
