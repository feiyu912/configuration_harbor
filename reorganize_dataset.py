import os
import json
import shutil
import random
from pathlib import Path

def log_info(message):
    """打印信息日志"""
    print(f"[INFO] {message}")

def log_error(message):
    """打印错误日志"""
    print(f"[ERROR] {message}")

def ensure_directory(path):
    """确保目录存在，如果不存在则创建"""
    os.makedirs(path, exist_ok=True)

def read_annotations(annotations_path):
    """读取标注文件"""
    try:
        with open(annotations_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        log_error(f"读取标注文件失败: {str(e)}")
        return None

def get_image_files(image_dir):
    """获取图像目录中的所有图像文件"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = []
    
    if not os.path.exists(image_dir):
        log_error(f"图像目录不存在: {image_dir}")
        return image_files
    
    for ext in image_extensions:
        for file in Path(image_dir).glob(f'*{ext}'):
            image_files.append(str(file))
    
    log_info(f"在 {image_dir} 中找到 {len(image_files)} 个图像文件")
    return image_files

def split_dataset(images, train_ratio=0.7, val_ratio=0.2):
    """
    分割数据集为训练集、验证集和测试集
    
    Args:
        images: 图像文件列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
    
    Returns:
        tuple: (训练集, 验证集, 测试集) 图像文件列表
    """
    # 打乱图像列表
    random.shuffle(images)
    
    total_count = len(images)
    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)
    
    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]
    
    log_info(f"数据集分割结果: 训练集 {len(train_images)} 张, 验证集 {len(val_images)} 张, 测试集 {len(test_images)} 张")
    
    return train_images, val_images, test_images

def create_yolo_annotations(annotations_data, image_file, output_dir):
    """
    根据COCO格式的标注创建YOLO格式的标注文件
    
    Args:
        annotations_data: COCO格式的标注数据
        image_file: 图像文件路径
        output_dir: 输出目录
    
    Returns:
        bool: 是否成功创建标注文件
    """
    # 获取图像文件名（不含扩展名）
    image_basename = Path(image_file).stem
    
    # 在标注数据中查找对应的图像信息
    image_id = None
    for img in annotations_data.get('images', []):
        if img.get('file_name') == Path(image_file).name:
            image_id = img.get('id')
            img_width = img.get('width')
            img_height = img.get('height')
            break
    
    if image_id is None:
        log_error(f"在标注数据中未找到图像: {Path(image_file).name}")
        return False
    
    # 查找该图像的所有标注
    image_annotations = []
    for ann in annotations_data.get('annotations', []):
        if ann.get('image_id') == image_id:
            image_annotations.append(ann)
    
    # 创建YOLO格式的标注文件
    output_path = os.path.join(output_dir, f"{image_basename}.txt")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for ann in image_annotations:
            category_id = ann.get('category_id', 0)
            
            # 处理边界框信息
            bbox = ann.get('bbox', [])
            if len(bbox) >= 4:
                x, y, width, height = bbox
                
                # 转换为YOLO格式：中心坐标和宽高归一化
                x_center = (x + width / 2) / img_width
                y_center = (y + height / 2) / img_height
                width_normalized = width / img_width
                height_normalized = height / img_height
                
                # 写入边界框信息
                f.write(f"{category_id} {x_center} {y_center} {width_normalized} {height_normalized}")
                
                # 如果有分割信息，也写入
                segmentation = ann.get('segmentation', [])
                if segmentation and isinstance(segmentation, list):
                    # 扁平化分割点列表
                    flat_points = []
                    for seg in segmentation:
                        if isinstance(seg, list):
                            # 归一化分割点
                            for i in range(0, len(seg), 2):
                                seg[i] = seg[i] / img_width  # x坐标归一化
                                seg[i+1] = seg[i+1] / img_height  # y坐标归一化
                            flat_points.extend(seg)
                    
                    # 将分割点添加到行尾
                    for point in flat_points:
                        f.write(f" {point}")
                
                f.write("\n")
    
    return True

def copy_files(file_list, source_dir, target_dir):
    """
    复制文件列表到目标目录
    
    Args:
        file_list: 文件路径列表
        source_dir: 源目录（用于提取相对路径）
        target_dir: 目标目录
    
    Returns:
        int: 成功复制的文件数
    """
    success_count = 0
    
    for file_path in file_list:
        try:
            # 获取相对路径
            relative_path = os.path.relpath(file_path, source_dir)
            # 构建目标路径
            target_path = os.path.join(target_dir, Path(file_path).name)
            # 复制文件
            shutil.copy2(file_path, target_path)
            success_count += 1
        except Exception as e:
            log_error(f"复制文件失败 {file_path}: {str(e)}")
    
    return success_count

def reorganize_dataset(base_dir="g:\configuration_harbor\harbor_port_backup"):
    """
    重新组织数据集，将总的图片和标注分类到train/val/test三个集合中
    
    Args:
        base_dir: 数据集基础目录
    """
    log_info("开始重新组织数据集...")
    
    # 定义路径
    annotations_path = os.path.join(base_dir, "annotations.json")
    total_images_dir = os.path.join(base_dir, "images")
    
    # 创建train/val/test目录结构
    split_dirs = {}
    for split in ['train', 'val', 'test']:
        split_dirs[split] = {
            'images': os.path.join(base_dir, split, 'images'),
            'labels': os.path.join(base_dir, split, 'labels')
        }
        ensure_directory(split_dirs[split]['images'])
        ensure_directory(split_dirs[split]['labels'])
    
    # 读取标注文件
    annotations_data = read_annotations(annotations_path)
    if not annotations_data:
        log_error("无法继续，标注文件读取失败")
        return False
    
    # 获取所有图像文件
    image_files = get_image_files(total_images_dir)
    if not image_files:
        log_error("无法继续，找不到图像文件")
        return False
    
    # 分割数据集
    train_images, val_images, test_images = split_dataset(image_files)
    
    # 处理每个分割集
    for split_name, images in [('train', train_images), ('val', val_images), ('test', test_images)]:
        log_info(f"\n处理 {split_name} 集 ({len(images)} 张图像)")
        
        # 复制图像文件
        log_info(f"正在复制图像文件到 {split_dirs[split_name]['images']}...")
        copied_count = copy_files(images, total_images_dir, split_dirs[split_name]['images'])
        log_info(f"成功复制 {copied_count} 个图像文件")
        
        # 创建YOLO格式的标注文件
        log_info(f"正在创建YOLO格式标注文件到 {split_dirs[split_name]['labels']}...")
        annotation_success_count = 0
        
        for image_file in images:
            if create_yolo_annotations(annotations_data, image_file, split_dirs[split_name]['labels']):
                annotation_success_count += 1
        
        log_info(f"成功创建 {annotation_success_count} 个标注文件")
    
    # 创建数据配置文件
    create_data_config(base_dir, split_dirs)
    
    log_info("\n数据集重新组织完成！")
    return True

def create_data_config(base_dir, split_dirs):
    """
    创建YOLO数据配置文件
    """
    # 获取类别信息
    categories = []
    annotations_path = os.path.join(base_dir, "annotations.json")
    
    try:
        with open(annotations_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for cat in data.get('categories', []):
            categories.append(cat.get('name', f'class_{cat.get('id', 0)}'))
    except Exception:
        # 如果无法获取类别信息，使用默认类别
        categories = ['ship', 'container', 'crane']
    
    # 创建配置内容
    config_content = f"""# Harbor Port Dataset Configuration
path: {base_dir}
train: train/images
val: val/images
test: test/images

# Classes
names:
{''.join([f'  {i}: {cat}\n' for i, cat in enumerate(categories)])}
"""
    
    # 写入配置文件
    config_path = os.path.join(base_dir, "data_config.yaml")
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    log_info(f"创建数据配置文件: {config_path}")

if __name__ == "__main__":
    reorganize_dataset()