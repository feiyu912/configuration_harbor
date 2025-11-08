# 数据集管理工具

这个目录包含了港口目标检测项目的数据集下载、验证和管理工具。

## 🚀 快速开始

### 1. 数据下载
```bash
python download_dataset.py
```

支持的数据来源：
- **Google Drive**: 自动从Google Drive下载完整数据集
- **OneDrive**: 从OneDrive直链下载  
- **手动下载**: 提供详细的手动下载和设置指导
- **示例数据**: 使用精简的示例数据进行快速体验

### 2. 数据验证
```bash
python verify_data.py
```

验证内容包括：
- ✅ 目录结构完整性
- ✅ 图像-标签配对关系
- ✅ YOLO标签格式正确性
- ✅ 配置文件有效性
- 📊 生成详细的验证报告

## 📋 手动数据管理指南

### 数据结构要求
```
configuration_harbor/
├── raw_public/
│   ├── images/          # 公开数据集图片
│   └── labels/          # 对应标签文件
├── raw_private/
│   ├── images/          # 私有数据集图片
│   └── labels/          # 对应标签文件
├── dataset_yolo_public/
│   ├── images/train/    # 训练集图片
│   ├── images/val/      # 验证集图片
│   ├── labels/train/    # 训练集标签
│   └── labels/val/      # 验证集标签
└── dataset_yolo_private/
    ├── images/train/
    ├── images/val/
    ├── labels/train/
    └── labels/val/
```

### 文件命名规范
- 图片文件：`.jpg`, `.png`, `.jpeg`
- 标签文件：`.txt`（YOLO格式）
- 文件名必须一一对应：`image.jpg` ↔ `image.txt`

### YOLO标签格式
每行一个目标，格式：`class_id x_center y_center width height`
- 所有坐标值归一化到 0-1 范围
- `class_id`: 0=ship, 1=container, 2=crane
- 坐标顺序：x_center, y_center, width, height

示例标签文件内容：
```
0 0.5 0.5 0.3 0.4    # ship
1 0.2 0.3 0.15 0.2   # container
2 0.8 0.7 0.25 0.3   # crane
```

## 🔧 配置说明

### 数据集配置 (`configs/`)
- `public_dataset.yaml` - 公开数据集配置
- `private_dataset.yaml` - 私有数据集配置
- `port.yaml` - 通用项目配置

### 关键参数
```yaml
train: dataset_yolo_public/images/train  # 训练图片路径
val: dataset_yolo_public/images/val      # 验证图片路径
nc: 3                                    # 类别数量
names: ['ship', 'container', 'crane']  # 类别名称
```

## 📊 数据验证检查项

### 1. 目录结构检查
- ✅ 必要目录是否存在
- ✅ 目录命名是否正确
- ✅ 子目录结构完整性

### 2. 文件配对检查
- ✅ 图片和标签文件数量匹配
- ✅ 文件名一一对应
- ✅ 无孤立文件（缺失配对）

### 3. 格式验证
- ✅ YOLO标签格式正确性
- ✅ 坐标值归一化范围（0-1）
- ✅ 类别ID有效性（0,1,2）
- ✅ 数值转换无错误

### 4. 配置验证
- ✅ YAML配置文件语法正确
- ✅ 必要字段完整性
- ✅ 类别数量与名称匹配

## 🎯 最佳实践建议

### 数据准备阶段
1. **小批量测试**：先用少量数据验证流程
2. **逐步扩展**：确认无误后再添加完整数据集
3. **备份原始数据**：处理前备份原始图片和标签

### 质量控制
1. **标签抽查**：人工检查部分标注框准确性
2. **格式统一**：确保所有文件格式一致
3. **命名规范**：使用统一的文件命名规则

### 性能优化
1. **图片尺寸**：推荐使用统一尺寸（如640x640）
2. **标签清理**：移除无效或错误的标注
3. **数据平衡**：确保各类别样本数量相对均衡

## 🚨 常见问题

### Q: 验证失败怎么办？
A: 根据验证报告逐一修复问题：
- 缺失目录 → 手动创建
- 配对失败 → 检查文件名或补充缺失文件
- 格式错误 → 修正YOLO标签格式
- 配置错误 → 更新YAML配置文件

### Q: 图片和标签数量不匹配？
A: 可能原因：
- 文件名不一致（大小写敏感）
- 文件扩展名不同
- 部分文件损坏或丢失

### Q: YOLO格式验证失败？
A: 常见错误：
- 坐标值超出0-1范围
- 类别ID不是整数
- 字段数量不是5个
- 数值转换错误

### Q: 如何处理大文件？
A: 推荐方案：
- 使用Git LFS跟踪大文件
- 云盘存储+下载脚本
- 分卷压缩上传
- 只上传必要文件到GitHub

## 🎉 下一步

数据验证通过后，你可以：

1. **查看展示系统**：`streamlit run app/streamlit_app.py`
2. **开始模型训练**：`python train.py --data configs/your_config.yaml`
3. **对比分析**：使用展示系统的val7 vs val8对比功能

访问 **http://localhost:8503** 体验完整的混合测试集深度分析！