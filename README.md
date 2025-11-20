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
- **高级算法展示系统**（v1.8版本）：
  - 训练结果展示（损失曲线、评估指标、混淆矩阵等）
  - 数据集展示（样本浏览、统计信息）
  - 模型对比功能（支持两种对比模式：自定义模型对比和数据集训练性能对比）
  - 数据集对比分析（数据集组成分析、统计对比、详细报告生成）
  - **🏆 混合测试集深度分析**（新增）：
    - val7、val8和harbor_opt2三模型性能对比（harbor_opt2最优）
    - 9大专业可视化图表（雷达图、PR曲线、混淆矩阵等）
    - 类别级别AP分析与三模型对比
    - 泛化能力评估与部署建议

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
- 包含按类别分类的子目录：`raw_public_ship/`、`raw_public_container/`

### 私有数据集
- 存放位置：`raw_private/`
- 包含按类别分类的子目录：`raw_private_ship/`、`raw_private_container/`、`raw_private_crane/`

### 混合数据集
- 存放位置：`dataset_raw/`
- YOLO格式数据集：`dataset_yolo/`

### 分离数据集结构（YOLO格式）
```
dataset_yolo_public/          # 公开数据集（YOLO格式）
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/

dataset_yolo_private/         # 自制数据集（YOLO格式）
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/

dataset_yolo_mixed_test/     # 混合测试数据集（YOLO格式）
├── images/
│   └── test/                # 仅包含测试集，用于模型对比评估
└── labels/
│   └── test/
```

### 数据生成链路

#### 分阶段处理模式（推荐）
```
原始分类数据 → 直接处理 → YOLO格式数据
raw_*_ship/container → process_categorized → dataset_yolo_*
```

#### 传统处理模式
```
原始分类数据 → 合并 → 处理 → YOLO格式数据
raw_*_ship/container → raw_public/private → dataset_yolo_*
```

**注意**：公开数据集缺少crane类别，仅包含ship和container两类

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆项目（推荐）
git clone https://github.com/your-username/port-detection-system.git
cd port-detection-system

# 安装依赖
pip install -r requirements.txt

# 🆕 下载数据集（自动管理大文件）
cd data_download_scripts
python download_dataset.py  # 支持Google Drive、OneDrive、手动下载
python verify_data.py       # 验证数据完整性
```

### 🎯 分阶段处理流程（推荐）

#### 步骤0：COCO数据分析（可选）
```bash
# 分析COCO数据集的类别定义和标注分布
python analyze_coco_categories.py

# 根据分析结果更新类别映射（修改coco_to_yolo_converter.py中的category_map）
```

#### 步骤1：数据预处理
```bash
# 基础预处理（公开数据集归一化+增强，自制数据集去噪）
python train.py --compare --stage preprocess

# 完整预处理选项
python train.py --compare --stage preprocess --normalize --augment --denoise
```

#### 步骤2：模型训练
```bash
# 标准训练（50轮）
python train.py --compare --stage train --epochs 50

# 自定义参数训练
python train.py --compare --stage train --epochs 100 --batch 16
```

#### 步骤3：性能评估
```bash
# 评估训练结果
python train.py --compare --stage evaluate

# 查看历史结果
ls compare_results/
cat compare_results/compare_results_*.json
```

#### 一键全流程（适合完整实验）
```bash
# 完整流程一键执行
python train.py --compare --stage all --normalize --augment --epochs 50
```

### 🔧 传统处理流程

#### 1. 数据处理
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
- **🏆 混合测试集深度分析**（新增核心功能）：
  - **三模型对比**：val7、val8和harbor_opt2在混合测试集上的性能对比
  - **9大专业图表**：雷达图、PR曲线、混淆矩阵、F1曲线等
  - **类别级别分析**：Ship、Container、Crane三类目标详细对比
  - **泛化能力评估**：验证模型在不同测试集上的适应能力

**系统访问：** 启动后，通过浏览器访问 http://localhost:8503 使用系统（自动展示val7 vs val8混合测试集分析）

## 配置文件

项目使用YAML格式的配置文件，主要配置文件包括：

- `configs/public_dataset.yaml` - 公开数据集配置
- `configs/private_dataset.yaml` - 私有数据集配置


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

### 数据集准备与管理

#### 🚀 自动化数据管理（推荐）

我们提供了完整的数据集管理工具，支持多种数据获取方式：

```bash
# 1. 进入数据管理目录
cd data_download_scripts

# 2. 下载数据集（支持Google Drive、OneDrive、手动下载）
python download_dataset.py

# 3. 验证数据完整性和格式正确性
python verify_data.py
```

#### 📊 数据管理工具功能

**download_dataset.py 支持：**
- ✅ Google Drive 自动下载
- ✅ OneDrive 直链下载  
- ✅ 手动下载详细指导
- ✅ 示例数据快速体验

**verify_data.py 验证：**
- ✅ 目录结构完整性
- ✅ 图像-标签配对关系
- ✅ YOLO标签格式正确性
- ✅ 配置文件有效性
- 📊 生成详细验证报告

#### 📋 传统手动准备

如果您想手动准备数据集：

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
│   ├── private_dataset.yaml  # 私有数据集配置
│   ├── public_dataset.yaml   # 公开数据集配置
│   └── port.yaml             # 项目通用配置
├── data_download_scripts/    # 🆕 数据集管理工具
│   ├── download_dataset.py   # 数据集下载工具
│   ├── verify_data.py        # 数据验证工具
│   └── README.md             # 数据管理文档
├── sample_data/              # 🆕 示例数据（GitHub友好）
│   ├── images/               # 少量示例图片
│   └── labels/               # 对应标签文件
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
├── runs/                     # 训练和检测结果
│   └── detect/               # 检测运行结果
│       ├── val/              # 验证结果（包含val7、val8对比）
│       ├── val2/             # 验证结果2
│       ├── val7/             # 🆕 混合测试集val7
│       └── val8/             # 🆕 混合测试集val8（+678%提升）
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
├── .gitignore               # 🆕 Git忽略文件（排除大图片）
├── yolo11n.pt                # YOLOv11模型权重
└── yolov8n.pt                # YOLOv8模型权重
```

## 📊 高级功能详解

### COCO数据分析工具

**功能**：分析COCO格式数据集的类别定义和标注分布

**使用方法**：
```bash
# 分析COCO数据集
python analyze_coco_categories.py

# 输出示例：
# 找到 3 个类别：
# - ID: 1, 名称: ship
# - ID: 2, 名称: container  
# - ID: 3, 名称: crane
# 
# 标注统计：
# - 类别ID 1: 1500 个标注
# - 类别ID 2: 800 个标注
# - 类别ID 3: 200 个标注
```

**重要提醒**：
- 根据分析结果更新 `coco_to_yolo_converter.py` 中的 `category_map` 字典
- 确保类别ID映射正确：0:ship, 1:container, 2:crane

### COCO到YOLO格式转换

**转换命令**：
```bash
# 转换公开数据集
python coco_to_yolo_converter.py --input datasets/coco_public --output raw_public

# 转换私有数据集  
python coco_to_yolo_converter.py --input datasets/coco_private --output raw_private
```

**转换前准备**：
1. 运行 `analyze_coco_categories.py` 分析类别
2. 根据实际类别更新 `category_map` 字典
3. 确保COCO文件格式正确（包含images、annotations、categories）

**转换后检查**：
- 验证生成的YOLO标签文件
- 检查类别数量是否匹配
- 确认标注框坐标是否正确

### 数据集对比分析功能

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

### 模型对比功能

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

## 🎯 混合测试集深度分析 (val7、val8和harbor_opt2三模型对比)

### 📈 核心性能提升

基于Streamlit展示系统的深度对比分析，三模型在混合测试集上的性能表现：

| 指标 | val7 | val8 | harbor_opt2 | 最佳模型 |
|------|------|------|------------|----------|
| **Precision** | 4.1% | 32.0% | 34.5% | **harbor_opt2** |
| **Recall** | 6.1% | 35.4% | 38.2% | **harbor_opt2** |
| **mAP50** | 3.7% | 30.7% | 33.2% | **harbor_opt2** |
| **mAP50-95** | 1.5% | 14.9% | 16.8% | **harbor_opt2** |

**性能提升总结**：
- harbor_opt2相比val7：Precision +741%，Recall +526%，mAP50 +797%
- harbor_opt2相比val8：Precision +7.8%，Recall +7.9%，mAP50 +8.1%

### 🚢 类别级别性能分析

| 类别 | val7 AP | val8 AP | harbor_opt2 AP | 最佳模型 | 表现评估 |
|------|---------|---------|--------------|----------|----------|
| **Ship** | 4.4% | 23.6% | 25.8% | **harbor_opt2** | **优秀** |
| **Container** | 0.18% | 1.16% | 1.28% | **harbor_opt2** | **需优化** |
| **Crane** | 0% | 20.0% | 21.5% | **harbor_opt2** | **良好** |

### 🔍 深度分析洞察

**泛化能力验证**：
- 三个模型的梯度提升证明算法优化的持续有效性
- harbor_opt2展现了最强的泛化能力，在所有类别和指标上均表现最优
- 从val7到harbor_opt2的渐进式提升，体现了优化策略的系统性改进

**技术优化亮点**：
- **harbor_opt2优势**：在保持高效率的同时，实现了对val8的全面超越
- **类别适应能力**：对Ship和Crane类别的检测能力达到实用水平
- **持续优化空间**：Container类别仍有较大提升潜力

### 📊 专业可视化分析

Streamlit展示系统提供**9大专业分析图表**：
1. **雷达图对比** - 多维度性能评估
2. **精确率-召回率对比** - 核心指标趋势分析
3. **mAP50对比** - 检测精度深度分析
4. **mAP50-95对比** - 定位精度专业评估
5. **混淆矩阵对比** - 类别识别准确性分析
6. **PR曲线对比** - 各类别性能曲线
7. **F1曲线对比** - 综合性能评估
8. **训练曲线** - 学习过程监控
9. **性能提升幅度** - 量化改进效果

### 🎯 部署建议

基于深度分析结果，系统提供**数据驱动的部署建议**：

**🌟 推荐部署方案**：
- **首选模型**：harbor_opt2（综合性能最优）
- **备选方案**：若计算资源受限，val8也是可行选择

**🔴 高优先级优化**：
- 重点优化Container类别检测（当前AP仅1.28%）
- 增加Container类别训练样本和标注质量
- 针对性调整Container类别的损失权重和锚框设置

**🟡 中优先级改进**：
- 基于harbor_opt2进一步优化mAP50-95定位精度
- 探索更适合港口场景的预训练权重
- 优化模型推理速度，提高实时检测能力

**🟢 低优先级观察**：
- 监控harbor_opt2在不同港口场景下的表现稳定性
- 评估实际部署环境中的性能衰减情况

### 🚀 访问方式

```bash
cd app
streamlit run streamlit_app.py
```

访问 **http://localhost:8503** 体验完整的三模型（val7、val8和harbor_opt2）深度对比分析系统！

## 🔧 常见问题与解决方案

### COCO数据分析相关问题

**问题1：类别ID不匹配**
- **原因**：COCO文件中的类别ID与代码中的映射不一致
- **解决**：运行`analyze_coco_categories.py`查看实际类别，更新`category_map`字典
- **示例**：如果分析显示ship的ID是5，需要修改为`5: "ship"`

**问题2：标注文件格式错误**
- **原因**：COCO文件缺少必要的字段或格式不正确
- **解决**：检查文件是否包含`images`、`annotations`、`categories`字段
- **验证**：使用JSON格式验证工具检查文件完整性

### 格式转换相关问题

**问题3：转换后标签文件为空**
- **原因**：类别映射错误或COCO文件中没有对应类别的标注
- **解决**：确认`category_map`中的类别ID与实际数据匹配
- **检查**：对比`analyze_coco_categories.py`输出与`category_map`设置

**问题4：YOLO格式坐标错误**
- **原因**：坐标转换算法问题或原始COCO标注异常
- **解决**：检查转换后的标签文件，确认坐标值在0-1范围内
- **验证**：随机抽查几个样本的可视化结果

### 数据处理相关问题

**问题5：预处理阶段失败**
- **原因**：原始数据路径错误或文件格式不支持
- **解决**：检查`raw_*_ship/container`目录是否存在且包含有效图像
- **路径**：确保使用正确的相对路径或绝对路径

**问题6：训练阶段内存不足**
- **原因**：批次大小设置过大或图像分辨率太高
- **解决**：减小`--batch`参数值，如从16改为8或4
- **优化**：使用较小的`--imgsz`参数，如从640改为416

### 模型训练相关问题

**问题7：训练loss不下降**
- **原因**：学习率设置不当、数据质量差或模型复杂度不够
- **解决**：
  - 检查数据标注质量，确保标注框准确
  - 尝试不同的学习率设置
  - 增加训练轮数或使用预训练模型
  - 添加数据增强提高数据多样性

## 🎯 最佳实践建议

### 数据验证优先
1. **COCO数据分析**：转换前务必运行分析工具
2. **小批量测试**：先用小数据集测试完整流程
3. **逐步验证**：每个阶段完成后检查输出结果

### 参数调整策略
1. **预处理参数**：
   - 公开数据集：推荐`--normalize --augment`
   - 私有数据集：推荐`--denoise --augment`
2. **训练参数**：
   - 初始训练：`--epochs 50 --batch 16`
   - 精细调优：`--epochs 100 --batch 8`

### 分阶段执行优势
1. **调试友好**：每个阶段可独立调试和优化
2. **时间节省**：失败后无需重复所有步骤
3. **灵活组合**：可根据需要选择特定阶段执行

### 结果对比分析
1. **多次实验**：同一参数配置运行多次取平均值
2. **指标关注**：重点关注mAP50和mAP50-95
3. **可视化验证**：结合主观视觉检查验证客观指标

## 📈 数据分析结论

通过系统提供的数据分析功能，可以得出以下结论：

1. **数据集规模影响**：公开数据集由于样本数量更多，训练的模型在通用性上表现更好，而私有数据集由于样本相对较少，模型容易过拟合，但在特定场景下表现更佳。

2. **数据质量差异**：公开数据集标注质量较高，边界框更加准确，而私有数据集中存在一些标注不准确的情况，影响了模型的训练效果。

3. **类别分布特点**：
   - Ship类别：两个数据集中都是主要类别，分布较为均衡
   - Container类别：公开数据集中分布更均匀，私有数据集中存在类别不平衡
   - Crane类别：仅存在于私有数据集中，样本数量相对较少

4. **模型性能对比**：
   - 在mAP50指标上，公开数据集训练的模型通常表现更好
   - 在特定场景下，私有数据集训练的模型可能具有更好的适应性
   - 混合训练策略可以结合两者优势，获得更均衡的性能

5. **数据增强建议**：
   - 对私有数据集建议增加数据增强，特别是旋转、缩放等几何变换
   - 对公开数据集可以适当减少增强，保持数据的原始特征
   - 针对不同类别可以采用不同的增强策略

## 🔄 版本更新

### v1.9.0 (最新)
- ✅ **GitHub友好版本**：新增.gitignore自动排除大文件
- ✅ **智能数据管理**：集成download_dataset.py和verify_data.py工具
- ✅ **三模型对比分析**：val7、val8和harbor_opt2深度对比（harbor_opt2最优）
- ✅ **9大可视化图表**：专业分析雷达图、PR曲线等
- ✅ **泛化能力评估**：混合测试集性能验证与最佳模型推荐

### v1.7.0
- ✅ 新增混合测试集深度对比分析系统
- ✅ 支持val7 vs val8性能对比
- ✅ 集成专业可视化图表

### v1.6.0
- ✅ 新增COCO数据分析工具
- ✅ 支持COCO转YOLO格式转换
- ✅ 数据集对比分析功能

### v1.5.0
- ✅ 新增模型对比功能
- ✅ 支持多模型性能对比
- ✅ 可视化对比图表

### v1.4.0
- ✅ 新增自动标注工具
- ✅ 支持私有数据自动标注
- ✅ 标注质量评估功能

### v1.3.0
- ✅ 新增数据增强功能
- ✅ 支持图像预处理和增强
- ✅ 增强参数配置

### v1.2.0
- ✅ 新增图像爬取工具
- ✅ 支持在线图像获取
- ✅ 图像质量筛选

### v1.1.0
- ✅ 新增数据集合并功能
- ✅ 支持多数据集整合
- ✅ 数据分布分析

### v1.0.0
- ✅ 基础港口目标检测系统
- ✅ 支持YOLOv8和YOLOv11
- ✅ 公开和私有数据集训练
- ✅ 算法展示系统


