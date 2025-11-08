#!/usr/bin/env python3
"""
数据集验证脚本
验证数据完整性、格式正确性和目录结构
"""

import os
import json
from pathlib import Path
import yaml

def verify_directory_structure():
    """验证目录结构"""
    print("🔍 验证目录结构...")
    
    required_dirs = [
        "raw_public/images", "raw_public/labels",
        "raw_private/images", "raw_private/labels", 
        "app", "configs", "src", "data"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"❌ 缺失目录: {missing_dirs}")
        return False
    else:
        print("✅ 目录结构验证通过")
        return True

def verify_image_label_pairs():
    """验证图像-标签配对"""
    print("🔍 验证图像-标签配对...")
    
    datasets = [
        ("raw_public/images", "raw_public/labels"),
        ("raw_private/images", "raw_private/labels"),
        ("dataset_yolo_public/images/train", "dataset_yolo_public/labels/train"),
        ("dataset_yolo_public/images/val", "dataset_yolo_public/labels/val"),
        ("dataset_yolo_private/images/train", "dataset_yolo_private/labels/train"),
        ("dataset_yolo_private/images/val", "dataset_yolo_private/labels/val")
    ]
    
    total_issues = 0
    for img_dir, lbl_dir in datasets:
        if os.path.exists(img_dir) and os.path.exists(lbl_dir):
            images = set(f.stem for f in Path(img_dir).glob("*") if f.suffix.lower() in ['.jpg', '.png', '.jpeg'])
            labels = set(f.stem for f in Path(lbl_dir).glob("*.txt"))
            
            missing_labels = images - labels
            missing_images = labels - images
            
            if missing_labels:
                print(f"⚠️  {img_dir}: {len(missing_labels)} 张图片缺少标签")
                total_issues += len(missing_labels)
            if missing_images:
                print(f"⚠️  {lbl_dir}: {len(missing_images)} 个标签缺少图片")
                total_issues += len(missing_images)
    
    if total_issues == 0:
        print("✅ 图像-标签配对验证通过")
        return True
    else:
        print(f"❌ 发现 {total_issues} 个配对问题")
        return False

def verify_yolo_labels():
    """验证YOLO标签格式"""
    print("🔍 验证YOLO标签格式...")
    
    label_dirs = [
        "dataset_yolo_public/labels/train",
        "dataset_yolo_public/labels/val", 
        "dataset_yolo_private/labels/train",
        "dataset_yolo_private/labels/val"
    ]
    
    total_issues = 0
    for lbl_dir in label_dirs:
        if os.path.exists(lbl_dir):
            for label_file in Path(lbl_dir).glob("*.txt"):
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                    
                    for line_num, line in enumerate(lines, 1):
                        parts = line.strip().split()
                        if len(parts) != 5:
                            print(f"❌ {label_file}:{line_num} - 格式错误: {len(parts)} 个字段")
                            total_issues += 1
                            continue
                        
                        try:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            # 验证归一化坐标
                            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                                   0 <= width <= 1 and 0 <= height <= 1):
                                print(f"⚠️  {label_file}:{line_num} - 坐标超出范围")
                                total_issues += 1
                                
                        except ValueError as e:
                            print(f"❌ {label_file}:{line_num} - 数值转换错误: {e}")
                            total_issues += 1
                            
                except Exception as e:
                    print(f"❌ 读取 {label_file} 失败: {e}")
                    total_issues += 1
    
    if total_issues == 0:
        print("✅ YOLO标签格式验证通过")
        return True
    else:
        print(f"❌ 发现 {total_issues} 个标签格式问题")
        return False

def verify_config_files():
    """验证配置文件"""
    print("🔍 验证配置文件...")
    
    config_files = [
        "configs/public_dataset.yaml",
        "configs/private_dataset.yaml", 
        "configs/port.yaml"
    ]
    
    total_issues = 0
    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                # 验证必要字段
                required_fields = ['nc', 'names']
                for field in required_fields:
                    if field not in config:
                        print(f"❌ {config_file} - 缺少必要字段: {field}")
                        total_issues += 1
                
                # 验证类别数量
                if 'nc' in config and 'names' in config:
                    if config['nc'] != len(config['names']):
                        print(f"❌ {config_file} - 类别数量不匹配: nc={config['nc']}, names={len(config['names'])}")
                        total_issues += 1
                        
            except Exception as e:
                print(f"❌ 读取 {config_file} 失败: {e}")
                total_issues += 1
        else:
            print(f"⚠️  配置文件不存在: {config_file}")
    
    if total_issues == 0:
        print("✅ 配置文件验证通过")
        return True
    else:
        print(f"❌ 发现 {total_issues} 个配置问题")
        return False

def generate_summary_report():
    """生成验证报告"""
    print("\n📊 验证总结报告")
    print("=" * 50)
    
    checks = [
        verify_directory_structure(),
        verify_image_label_pairs(), 
        verify_yolo_labels(),
        verify_config_files()
    ]
    
    passed = sum(checks)
    total = len(checks)
    
    print(f"\n验证结果: {passed}/{total} 项通过")
    
    if passed == total:
        print("🎉 数据集验证完全通过！可以开始训练")
        print("\n下一步操作:")
        print("1. 运行 'streamlit run app/streamlit_app.py' 查看展示系统")
        print("2. 运行 'python train.py --data configs/your_config.yaml' 开始训练")
    else:
        print("⚠️  发现一些问题，建议修复后再进行训练")
        print("\n修复建议:")
        print("1. 检查缺失的文件和目录")
        print("2. 验证图像-标签配对关系") 
        print("3. 检查YOLO标签格式")
        print("4. 确认配置文件正确性")

def main():
    """主函数"""
    print("🔍 港口目标检测数据集验证工具")
    print("=" * 50)
    
    generate_summary_report()

if __name__ == "__main__":
    main()