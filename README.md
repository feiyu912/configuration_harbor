# 港口目标检测项目（增强版）

这个项目使用YOLO模型进行港口场景中的目标检测，支持检测船舶（ship）、集装箱（container）和起重机（crane）三类目标。本增强版添加了完整的数据预处理和增强功能，以及高级的算法展示系统，支持数据集对比分析、模型性能对比和可视化展示。

## 功能特点

- 支持分开训练公开数据集和自制数据集
- 提供数据集比较功能，评估不同数据集的训练效果
- 支持使用混合数据集进行训练
- 包含完整的数据处理、训练和部署流程
- **增强的数据预处理功能**：
  - 图像归一化处理
  - 多种图像去噪方法（高斯滤波、中值滤波、双边滤波）
  - 标签验证和修复功能
  - 缺失标签自动生成
- **丰富的数据增强操作**：
  - 水平翻转
  - 随机旋转（-15°到15°）
  - 亮度和对比度调整
  - 高斯噪声添加
- **高级算法展示系统**（v1.1版本）：
  - 训练结果展示（损失曲线、评估指标、混淆矩阵等）
  - 数据集展示（样本浏览、统计信息）
  - 模型对比功能（支持两种对比模式：自定义模型对比和数据集训练性能对比）
  - 数据集对比分析（数据集组成分析、统计对比、详细报告生成）

## 环境要求

在开始前，请确保您已安装以下依赖：

- Python 3.8+
- CUDA 11.3+（如果使用GPU训练）
- 主要依赖库在 `requirements.txt` 文件中列出

## 安装

1. 克隆或下载项目代码
2. 安装依赖：

```bash
pip install -r requirements.txt
```

## 数据集结构

本项目支持使用以下数据集进行训练和测试：

### 公开数据集
- 存放位置：`raw_public/`
- 包含按类别分类的子目录：`raw_public_ship/`、`raw_public_container/`、`raw_public_crane/`

### 私有数据集
- 存放位置：`raw_private/`
- 包含按类别分类的子目录：`raw_private_ship/`、`raw_private_container/`、`raw_private_crane/`

### 混合数据集
- 存放位置：`dataset_raw/`
- YOLO格式数据集：`dataset_yolo/`

## 快速开始

### 数据处理

```bash
# 预处理并合并数据集
python merge_datasets.py

# 增强数据处理
python enhanced_data_processor.py
```

### 模型训练

训练命令示例：

```bash
# 使用公开数据集训练
python train.py --data configs/public_dataset.yaml --weights yolov8n.pt

# 使用私有数据集训练
python train.py --data configs/private_dataset.yaml --weights yolov8n.pt

# 使用混合数据集训练
python train.py --data configs/mixed_dataset.yaml --weights yolov8n.pt
```

### 模型推理

```bash
# 使用训练好的模型进行推理
python src/infer.py --model runs/train/your_model/weights/best.pt --source test_images/
```

### 启动算法展示系统

```bash
cd app
streamlit run streamlit_app.py
```

**系统主要功能：**
- **训练结果展示**：展示各个训练模型的损失曲线、评估指标、混淆矩阵等
- **数据集展示**：浏览数据集样本，查看数据集统计信息
- **模型对比**：对比不同模型的性能指标，支持雷达图展示
- **数据集对比分析**：分析和对比不同数据集的组成特点和训练性能

**系统访问：** 启动后，通过浏览器访问 http://localhost:8501 使用系统

## 配置文件

项目使用YAML格式的配置文件，主要配置文件包括：

- `configs/public_dataset.yaml` - 公开数据集配置
- `configs/private_dataset.yaml` - 私有数据集配置
- `configs/mixed_dataset.yaml` - 混合数据集配置

配置文件中包含的主要参数：

```yaml
train: dataset_yolo/images/train
val: dataset_yolo/images/val

# 类别数量和名称
nc: 3
names: ['ship', 'container', 'crane']

# 训练参数
train_batch_size: 16
val_batch_size: 16
epochs: 100
image_size: 640

# 数据增强参数
augmentations:
  h_flip: true
  rotate: true
  brightness_contrast: true
  noise: true
```

## 数据处理参数说明

### enhanced_data_processor.py 可选参数

- `--input_dir`：输入图像目录
- `--output_dir`：输出处理后图像目录
- `--label_dir`：标签目录
- `--denoise`：应用去噪（可选：gaussian, median, bilateral）
- `--normalize`：应用归一化
- `--fix_labels`：验证和修复标签
- `--gen_missing`：为缺失标签的图像生成标签

## 数据集准备

如果您想使用自己的数据集，请按照以下步骤准备：

1. 创建图像和标签文件夹
2. 图像文件格式：.jpg、.png等
3. 标签格式：YOLO格式（class_id x_center y_center width height，归一化到0-1范围）
4. 图像和标签文件命名保持一致

## 模型推理

推理命令示例：

```bash
python src/infer.py --model runs/train/your_model/weights/best.pt --source test_images/ --conf 0.3 --iou 0.45
```

主要参数：
- `--model`：模型权重文件路径
- `--source`：图像/视频源或目录
- `--conf`：置信度阈值
- `--iou`：IoU阈值
- `--output`：输出结果保存目录

## 项目结构

```
├── app/                      # 算法展示系统
│   └── streamlit_app.py      # Streamlit应用主程序
├── configs/                  # 配置文件目录
│   ├── mixed_dataset.yaml    # 混合数据集配置
│   ├── private_dataset.yaml  # 私有数据集配置
│   ├── public_dataset.yaml   # 公开数据集配置
│   └── port.yaml             # 项目通用配置
├── dataset_raw/              # 原始混合数据集
│   ├── images/               # 图像文件
│   └── labels/               # 标签文件
├── dataset_yolo/             # YOLO格式数据集
│   ├── images/               # 训练、验证、测试图像
│   └── labels/               # YOLO格式标签
├── raw_private/              # 私有数据集
│   ├── images/               # 私有数据集图像
│   └── labels/               # 私有数据集标签
├── raw_public/               # 公开数据集
│   ├── images/               # 公开数据集图像
│   └── labels/               # 公开数据集标签
├── raw_private_ship/         # 私有船舶数据集（按类别分类）
├── raw_private_container/    # 私有集装箱数据集（按类别分类）
├── raw_private_crane/        # 私有起重机数据集（按类别分类）
├── raw_public_ship/          # 公开船舶数据集（按类别分类）
├── raw_public_container/     # 公开集装箱数据集（按类别分类）
├── raw_public_crane/         # 公开起重机数据集（按类别分类）
├── runs/                     # 训练和检测结果
│   └── detect/               # 检测运行结果
│       ├── mixed_dataset/    # 混合数据集训练结果
│       ├── port_custom/      # 自定义训练结果
│       ├── port_private/     # 私有数据集训练结果
│       ├── port_public/      # 公开数据集训练结果
├── src/                      # 源代码目录
│   ├── api.py                # API服务
│   ├── convert_voc_to_yolo.py # VOC转YOLO格式
│   ├── data_prep.py          # 数据准备脚本
│   ├── get_dataset.py        # 数据集获取脚本
│   ├── infer.py              # 推理脚本
│   └── train.py              # 训练核心模块
├── enhanced_data_processor.py # 增强版数据处理工具
├── auto_private_annotate.py  # 自动标注工具
├── crane_crawler.py          # 起重机图像爬取工具
├── merge_datasets.py         # 数据集合并工具
├── train.py                  # 训练主脚本
├── requirements.txt          # 依赖列表
├── yolo11n.pt                # YOLOv11模型权重
└── yolov8n.pt                # YOLOv8模型权重
```

## 数据集对比分析功能

算法展示系统的数据集对比分析功能可以帮助用户深入了解不同数据集的特点：

1. **数据集组成对比**：
   - 自动分析私有数据集、公开数据集和混合数据集
   - 生成包含图像数量、标签数量、目标数量的对比图表
   - 展示各类别分布的堆叠柱状图和饼图

2. **统计分析报告**：
   - 自动生成详细的数据分析报告
   - 包含数据规模对比、目标密度对比和类别分布特点分析
   - 提供数据增强策略和训练策略建议

3. **可视化展示**：
   - 使用多种图表类型直观展示对比结果
   - 支持展开查看各个数据集的详细信息
   - 提供交互式的数据浏览体验

## 模型对比功能

算法展示系统提供两种模型对比模式：

1. **自定义模型对比**：
   - 选择任意训练结果进行对比
   - 支持多个评估指标的对比展示
   - 自动标记最佳性能值和对应的训练轮次
   - 提供详细的性能指标对比表格

2. **数据集训练性能对比**：
   - 自动识别不同数据集的最佳训练结果
   - 对比不同数据集训练模型的性能差异
   - 提供雷达图多维度展示模型性能
   - 帮助分析数据集对模型性能的影响

## 数据分析结论

通过系统提供的数据分析功能，可以得出以下结论：

1. **数据集互补性**：私有数据集和公开数据集在目标密度、类别分布等方面存在差异，混合使用可以提高模型的泛化能力。

2. **数据增强策略建议**：
   - 对于类别分布不平衡的数据集，建议采用过采样或类别权重调整策略
   - 对于目标密度较低的数据集，可以考虑增加合成样本

3. **训练策略优化**：
   - 利用混合数据集进行训练，同时保留对私有数据集的微调阶段
   - 针对不同数据集的特点，调整数据增强参数
   - 采用交叉验证评估模型在不同数据分布下的鲁棒性

## 版本更新记录

### v1.1 版本更新（最新）
- 添加数据集对比分析功能
- 增强数据分析的严谨性和可视化效果
- 改进模型对比功能，支持雷达图展示
- 优化整体用户界面和交互体验

### v1.0 版本
- 初始版本发布
- 基本的数据处理和训练功能
- 简单的可视化界面


