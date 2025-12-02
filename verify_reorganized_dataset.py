import os
import glob
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

def verify_split(split_name, images_dir, labels_dir):
    """
    验证单个数据集分割的完整性
    
    Args:
        split_name: 分割名称 (train/val/test)
        images_dir: 图像目录
        labels_dir: 标注目录
    
    Returns:
        dict: 验证结果
    """
    log_info(f"验证 {split_name} 集...")
    
    results = {
        'total_images': 0,
        'total_labels': 0,
        'missing_labels': [],
        'unmatched_labels': []
    }
    
    # 检查目录是否存在
    if not os.path.exists(images_dir):
        log_error(f"{split_name} 图像目录不存在: {images_dir}")
        return results
    
    if not os.path.exists(labels_dir):
        log_error(f"{split_name} 标注目录不存在: {labels_dir}")
        return results
    
    # 获取所有图像文件
    image_extensions = get_image_extensions()
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, f'*{ext}')))
    
    results['total_images'] = len(image_files)
    log_info(f"{split_name} 集包含 {len(image_files)} 个图像文件")
    
    # 获取所有标注文件
    label_files = glob.glob(os.path.join(labels_dir, '*.txt'))
    results['total_labels'] = len(label_files)
    log_info(f"{split_name} 集包含 {len(label_files)} 个标注文件")
    
    # 创建文件名映射（不包含扩展名）
    image_basenames = {Path(img).stem: img for img in image_files}
    label_basenames = {Path(label).stem: label for label in label_files}
    
    # 检查缺失的标注文件
    for img_basename, img_path in image_basenames.items():
        if img_basename not in label_basenames:
            missing_label = os.path.join(labels_dir, f"{img_basename}.txt")
            results['missing_labels'].append({
                'image': img_path,
                'expected_label': missing_label
            })
            log_error(f"{split_name} 集缺失标注文件: {img_path} -> {missing_label}")
    
    # 检查不匹配的标注文件
    for label_basename, label_path in label_basenames.items():
        if label_basename not in image_basenames:
            results['unmatched_labels'].append({
                'label': label_path
            })
            log_error(f"{split_name} 集不匹配标注文件: {label_path}")
    
    # 打印验证结果摘要
    log_info(f"{split_name} 集验证结果: 图像 {results['total_images']}, 标注 {results['total_labels']}")
    log_info(f"  缺失标注数: {len(results['missing_labels'])}")
    log_info(f"  不匹配标注数: {len(results['unmatched_labels'])}")
    
    if len(results['missing_labels']) == 0 and len(results['unmatched_labels']) == 0:
        log_info(f"  ✓ {split_name} 集验证通过！")
    else:
        log_info(f"  ✗ {split_name} 集验证失败！")
    
    return results

def verify_dataset(base_dir="g:\configuration_harbor\harbor_port_backup"):
    """
    验证重组后的数据集
    
    Args:
        base_dir: 数据集基础目录
    """
    log_info(f"开始验证重组后的数据集: {base_dir}")
    
    splits = ['train', 'val', 'test']
    all_results = {}
    
    for split in splits:
        images_dir = os.path.join(base_dir, split, 'images')
        labels_dir = os.path.join(base_dir, split, 'labels')
        all_results[split] = verify_split(split, images_dir, labels_dir)
    
    # 生成总体验证报告
    log_info("\n=== 总体验证报告 ===")
    total_images = sum(r['total_images'] for r in all_results.values())
    total_labels = sum(r['total_labels'] for r in all_results.values())
    total_missing = sum(len(r['missing_labels']) for r in all_results.values())
    total_unmatched = sum(len(r['unmatched_labels']) for r in all_results.values())
    
    log_info(f"总图像数: {total_images}")
    log_info(f"总标注数: {total_labels}")
    log_info(f"总缺失标注数: {total_missing}")
    log_info(f"总不匹配标注数: {total_unmatched}")
    
    if total_missing == 0 and total_unmatched == 0:
        log_info("✓ 数据集验证通过！所有图像都有对应的标注文件。")
        return True
    else:
        log_info("✗ 数据集验证失败！请检查上述错误。")
        return False

if __name__ == "__main__":
    verify_dataset()