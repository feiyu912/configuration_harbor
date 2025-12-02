import os
import glob
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("segmentation_validation.log"),
        logging.StreamHandler()
    ]
)

def log_info(message):
    """打印信息日志"""
    logging.info(message)
    print(f"[INFO] {message}")

def log_error(message):
    """打印错误日志"""
    logging.error(message)
    print(f"[ERROR] {message}")

def log_warning(message):
    """打印警告日志"""
    logging.warning(message)
    print(f"[WARNING] {message}")

def validate_segmentation_annotation(label_path, max_classes=3):
    """
    验证YOLO分割标注文件的有效性
    
    Args:
        label_path: 标注文件路径
        max_classes: 最大类别ID (默认3个类别: 0, 1, 2)
    
    Returns:
        tuple: (是否有效, 错误信息列表, 警告信息列表)
    """
    is_valid = True
    errors = []
    warnings = []
    
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            errors.append("标注文件为空")
            is_valid = False
            return is_valid, errors, warnings
        
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                warnings.append(f"第{line_idx+1}行: 空行")
                continue
            
            parts = line.split()
            if len(parts) < 9:  # 1个类别 + 至少8个坐标点(4个点的多边形)
                errors.append(f"第{line_idx+1}行: 标注格式错误，数据不足 (需要至少1个类别+8个坐标点)")
                is_valid = False
                continue
            
            # 检查类别ID
            try:
                class_id = int(parts[0])
                if class_id < 0 or class_id >= max_classes:
                    errors.append(f"第{line_idx+1}行: 类别ID {class_id} 无效 (应为0-{max_classes-1})")
                    is_valid = False
            except ValueError:
                errors.append(f"第{line_idx+1}行: 类别ID '{parts[0]}' 不是有效的整数")
                is_valid = False
                continue
            
            # 检查坐标点是否为偶数个（每个点包含x和y）
            coordinates = parts[1:]
            if len(coordinates) % 2 != 0:
                errors.append(f"第{line_idx+1}行: 坐标点数量为奇数，应该成对出现")
                is_valid = False
                continue
            
            # 检查坐标值是否在[0,1]范围内
            for coord_idx, coord in enumerate(coordinates):
                try:
                    coord_value = float(coord)
                    if coord_value < 0 or coord_value > 1:
                        errors.append(f"第{line_idx+1}行: 坐标值 {coord_value} 超出范围 [0,1]")
                        is_valid = False
                except ValueError:
                    errors.append(f"第{line_idx+1}行: 坐标值 '{coord}' 不是有效的浮点数")
                    is_valid = False
                    continue
            
            # 检查多边形点数量是否合理
            num_points = len(coordinates) // 2
            if num_points < 3:
                errors.append(f"第{line_idx+1}行: 多边形点数不足3个")
                is_valid = False
            elif num_points > 1000:  # 防止异常大的多边形
                warnings.append(f"第{line_idx+1}行: 多边形点数过多 ({num_points}个)")
    
    except Exception as e:
        errors.append(f"读取文件时出错: {str(e)}")
        is_valid = False
    
    return is_valid, errors, warnings

def validate_dataset_segmentation(base_dir="g:\configuration_harbor\harbor_port_backup"):
    """
    验证数据集的所有分割标注文件
    
    Args:
        base_dir: 数据集基础目录
    """
    log_info(f"开始验证分割标注文件: {base_dir}")
    
    splits = ['train', 'val', 'test']
    results = {
        'total_files': 0,
        'valid_files': 0,
        'invalid_files': 0,
        'total_errors': 0,
        'total_warnings': 0,
        'errors_by_file': {},
        'warnings_by_file': {}
    }
    
    for split in splits:
        labels_dir = os.path.join(base_dir, split, 'labels')
        log_info(f"验证 {split} 集的标注文件...")
        
        if not os.path.exists(labels_dir):
            log_error(f"标注目录不存在: {labels_dir}")
            continue
        
        label_files = glob.glob(os.path.join(labels_dir, '*.txt'))
        log_info(f"在 {split} 集中找到 {len(label_files)} 个标注文件")
        
        for label_path in label_files:
            results['total_files'] += 1
            relative_path = os.path.relpath(label_path, base_dir)
            
            is_valid, errors, warnings = validate_segmentation_annotation(label_path)
            
            if errors:
                results['invalid_files'] += 1
                results['total_errors'] += len(errors)
                results['errors_by_file'][relative_path] = errors
                log_error(f"文件 {relative_path} 无效:")
                for error in errors:
                    log_error(f"  - {error}")
            
            if warnings:
                results['total_warnings'] += len(warnings)
                results['warnings_by_file'][relative_path] = warnings
                log_warning(f"文件 {relative_path} 警告:")
                for warning in warnings:
                    log_warning(f"  - {warning}")
            
            if is_valid and not errors:
                results['valid_files'] += 1
                log_info(f"文件 {relative_path} 验证通过")
    
    # 生成总结报告
    log_info("\n=== 分割标注验证总结 ===")
    log_info(f"总标注文件数: {results['total_files']}")
    log_info(f"有效文件数: {results['valid_files']}")
    log_info(f"无效文件数: {results['invalid_files']}")
    log_info(f"总错误数: {results['total_errors']}")
    log_info(f"总警告数: {results['total_warnings']}")
    
    # 输出错误文件详情
    if results['invalid_files'] > 0:
        log_info("\n=== 无效文件详情 ===")
        for file_path, file_errors in results['errors_by_file'].items():
            log_info(f"文件: {file_path}")
            for error in file_errors:
                log_info(f"  - {error}")
    
    # 输出警告文件详情
    if results['total_warnings'] > 0:
        log_info("\n=== 警告文件详情 ===")
        for file_path, file_warnings in results['warnings_by_file'].items():
            log_info(f"文件: {file_path}")
            for warning in file_warnings:
                log_info(f"  - {warning}")
    
    return results

def main():
    results = validate_dataset_segmentation()
    
    # 保存结果到文件
    with open("segmentation_validation_results.txt", "w", encoding="utf-8") as f:
        f.write("分割标注验证结果\n")
        f.write("="*50 + "\n\n")
        f.write(f"总标注文件数: {results['total_files']}\n")
        f.write(f"有效文件数: {results['valid_files']}\n")
        f.write(f"无效文件数: {results['invalid_files']}\n")
        f.write(f"总错误数: {results['total_errors']}\n")
        f.write(f"总警告数: {results['total_warnings']}\n\n")
        
        if results['invalid_files'] > 0:
            f.write("无效文件详情:\n")
            f.write("-"*50 + "\n")
            for file_path, errors in results['errors_by_file'].items():
                f.write(f"文件: {file_path}\n")
                for error in errors:
                    f.write(f"  - {error}\n")
                f.write("\n")
    
    log_info(f"验证结果已保存到 segmentation_validation_results.txt")

if __name__ == "__main__":
    main()