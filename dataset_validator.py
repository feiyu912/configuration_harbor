import os
import glob
import json
from pathlib import Path

def log_info(message):
    """打印信息日志"""
    print(f"[INFO] {message}")

def log_error(message):
    """打印错误日志"""
    print(f"[ERROR] {message}")

def get_image_extensions():
    """返回常见的图像文件扩展名"""
    return ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

def validate_dataset_split(image_dir, label_dir):
    """
    验证单个数据集分割（训练集/验证集/测试集）的图像和标注匹配情况
    
    Args:
        image_dir: 图像文件夹路径
        label_dir: 标注文件夹路径
    
    Returns:
        dict: 包含验证结果的字典
    """
    results = {
        'total_images': 0,
        'total_labels': 0,
        'missing_labels': [],  # 有图像但无标注的文件
        'unmatched_labels': [],  # 有标注但无图像的文件
        'label_format_errors': []  # 标注格式错误的文件
    }
    
    # 检查目录是否存在
    if not os.path.exists(image_dir):
        log_error(f"图像目录不存在: {image_dir}")
        return results
    
    if not os.path.exists(label_dir):
        log_error(f"标注目录不存在: {label_dir}")
        return results
    
    # 获取所有图像文件
    image_extensions = get_image_extensions()
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, f'*{ext}')))
    
    results['total_images'] = len(image_files)
    log_info(f"找到 {len(image_files)} 个图像文件")
    
    # 获取所有标注文件
    label_files = glob.glob(os.path.join(label_dir, '*.txt'))
    results['total_labels'] = len(label_files)
    log_info(f"找到 {len(label_files)} 个标注文件")
    
    # 创建文件名映射（不包含扩展名）
    image_basenames = {Path(img).stem: img for img in image_files}
    label_basenames = {Path(label).stem: label for label in label_files}
    
    # 检查缺失的标注文件
    for img_basename, img_path in image_basenames.items():
        if img_basename not in label_basenames:
            missing_label = os.path.join(label_dir, f"{img_basename}.txt")
            results['missing_labels'].append({
                'image': img_path,
                'expected_label': missing_label
            })
            log_error(f"缺失标注文件: 图像 {img_path} 没有对应的标注文件")
    
    # 检查不匹配的标注文件（有标注但无图像）
    for label_basename, label_path in label_basenames.items():
        if label_basename not in image_basenames:
            results['unmatched_labels'].append({
                'label': label_path,
                'missing_image': None  # 可以扩展为查找相似名称的图像
            })
            log_error(f"不匹配的标注文件: 标注 {label_path} 没有对应的图像文件")
    
    # 检查标注文件格式（可选）
    for label_basename, label_path in label_basenames.items():
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line_idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # YOLO格式: class_id x_center y_center width height [segmentation_points...]
                parts = line.split()
                if len(parts) < 5:
                    raise ValueError(f"行 {line_idx + 1}: 格式错误，至少需要5个值")
                
                # 检查类别ID是否为整数
                try:
                    int(parts[0])
                except ValueError:
                    raise ValueError(f"行 {line_idx + 1}: 类别ID不是整数")
                
                # 检查坐标值是否在0-1范围内
                coords = list(map(float, parts[1:]))
                for i, coord in enumerate(coords):
                    if coord < 0 or coord > 1:
                        raise ValueError(f"行 {line_idx + 1}: 坐标值 {coord} 不在0-1范围内")
                    
        except Exception as e:
            results['label_format_errors'].append({
                'label': label_path,
                'error': str(e)
            })
            log_error(f"标注格式错误: {label_path} - {str(e)}")
    
    return results

def validate_harbor_dataset(base_dir="g:\configuration_harbor\harbor_port_backup"):
    """
    验证harbor_port_backup数据集
    
    Args:
        base_dir: 数据集基础目录
    """
    log_info(f"开始验证数据集: {base_dir}")
    
    splits = ['train', 'val', 'test']
    all_results = {}
    
    for split in splits:
        log_info(f"\n验证 {split} 集...")
        image_dir = os.path.join(base_dir, split, 'images')
        label_dir = os.path.join(base_dir, split, 'labels')
        
        results = validate_dataset_split(image_dir, label_dir)
        all_results[split] = results
        
        # 打印摘要
        log_info(f"\n{split} 集验证结果:")
        log_info(f"  总图像数: {results['total_images']}")
        log_info(f"  总标注数: {results['total_labels']}")
        log_info(f"  缺失标注数: {len(results['missing_labels'])}")
        log_info(f"  不匹配标注数: {len(results['unmatched_labels'])}")
        log_info(f"  标注格式错误数: {len(results['label_format_errors'])}")
    
    # 保存结果到文件
    output_file = os.path.join(os.path.dirname(base_dir), 'dataset_validation_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    log_info(f"\n验证结果已保存到: {output_file}")
    
    # 生成总体摘要
    total_images = sum(r['total_images'] for r in all_results.values())
    total_labels = sum(r['total_labels'] for r in all_results.values())
    total_missing = sum(len(r['missing_labels']) for r in all_results.values())
    total_unmatched = sum(len(r['unmatched_labels']) for r in all_results.values())
    total_format_errors = sum(len(r['label_format_errors']) for r in all_results.values())
    
    log_info("\n=== 总体验证结果 ===")
    log_info(f"总图像数: {total_images}")
    log_info(f"总标注数: {total_labels}")
    log_info(f"缺失标注数: {total_missing}")
    log_info(f"不匹配标注数: {total_unmatched}")
    log_info(f"标注格式错误数: {total_format_errors}")
    
    if total_missing == 0 and total_unmatched == 0 and total_format_errors == 0:
        log_info("\n✓ 数据集验证通过！图像和标注文件完全匹配。")
    else:
        log_info("\n✗ 数据集验证失败！请检查上述错误。")
    
    return all_results

if __name__ == '__main__':
    validate_harbor_dataset()