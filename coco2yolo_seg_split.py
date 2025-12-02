import json
import os
import argparse
import random
from collections import defaultdict


def convert_coco_to_yolo_seg_with_split(coco_json_path, output_base_dir, split_ratios=None, images_source_dir=None):
    """
    将COCO格式的多边形标注转换为YOLOv8-seg格式，并按train/val/test划分输出
    
    Args:
        coco_json_path: COCO格式的JSON文件路径
        output_base_dir: 输出基础目录（会创建train/val/test子目录）
        split_ratios: 划分比例，默认[0.7, 0.2, 0.1] (train/val/test)
        images_source_dir: 图片源目录，如果提供则会复制图片到对应目录
    """
    if split_ratios is None:
        split_ratios = [0.7, 0.2, 0.1]  # train/val/test
    
    # 创建输出目录
    output_dirs = {
        'train': os.path.join(output_base_dir, 'train', 'labels'),
        'val': os.path.join(output_base_dir, 'val', 'labels'),
        'test': os.path.join(output_base_dir, 'test', 'labels')
    }
    
    # 创建图片输出目录
    image_output_dirs = {
        'train': os.path.join(output_base_dir, 'train', 'images'),
        'val': os.path.join(output_base_dir, 'val', 'images'),
        'test': os.path.join(output_base_dir, 'test', 'images')
    }
    
    for dir_path in list(output_dirs.values()) + list(image_output_dirs.values()):
        os.makedirs(dir_path, exist_ok=True)
    
    # 读取COCO JSON文件
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # ✅ 5. 构建类别映射，保持字典序，防止ID跳号
    wanted_ids = [1, 2, 3]  # 我们需要的COCO类别ID
    category_mapping = {coco_id: idx for idx, coco_id in enumerate(sorted(wanted_ids))}
    
    # 验证并打印类别映射
    for cat in data['categories']:
        coco_id = cat['id']
        if coco_id in category_mapping:
            yolo_id = category_mapping[coco_id]
            print(f"类别映射: COCO ID {coco_id} ({cat['name']}) -> YOLO ID {yolo_id}")
        else:
            print(f"警告: 类别ID {coco_id} ({cat['name']}) 不在目标范围内，将被跳过")
    
    # 构建图像ID到图像信息的映射
    image_id_to_info = {}
    for img in data['images']:
        image_id_to_info[img['id']] = img
    
    # 按图像分组标注
    annotations_by_image = defaultdict(list)
    for ann in data['annotations']:
        image_id = ann['image_id']
        annotations_by_image[image_id].append(ann)
    
    # 获取所有图像ID
    all_image_ids = list(annotations_by_image.keys())
    total_images = len(all_image_ids)
    
    # 随机划分数据集
    random.shuffle(all_image_ids)
    
    train_split = int(total_images * split_ratios[0])
    val_split = train_split + int(total_images * split_ratios[1])
    
    train_ids = set(all_image_ids[:train_split])
    val_ids = set(all_image_ids[train_split:val_split])
    test_ids = set(all_image_ids[val_split:])
    
    print(f"数据集划分:")
    print(f"训练集: {len(train_ids)} 张图片")
    print(f"验证集: {len(val_ids)} 张图片")
    print(f"测试集: {len(test_ids)} 张图片")
    
    # 统计信息
    total_annotations = len(data['annotations'])
    processed_annotations = 0
    skipped_annotations = 0
    
    # 处理每个划分的数据
    for split_name, image_ids in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
        split_annotations = 0
        
        for image_id in image_ids:
            if image_id not in image_id_to_info:
                print(f"警告: 图像ID {image_id} 未在images列表中找到")
                continue
            
            image_info = image_id_to_info[image_id]
            image_width = image_info['width']
            image_height = image_info['height']
            image_filename = image_info['file_name']
            annotations = annotations_by_image[image_id]
            
            # 创建YOLO格式的标签文件
            label_filename = os.path.splitext(image_filename)[0] + '.txt'
            label_path = os.path.join(output_dirs[split_name], label_filename)
            
            with open(label_path, 'w', encoding='utf-8') as f:
                for ann in annotations:
                    # 跳过crowd标注（RLE格式）
                    if ann.get('iscrowd', 0) == 1:
                        skipped_annotations += 1
                        continue
                    
                    # 获取类别ID
                    coco_category_id = ann['category_id']
                    if coco_category_id not in category_mapping:
                        print(f"警告: 类别ID {coco_category_id} 未在categories列表中找到")
                        skipped_annotations += 1
                        continue
                    
                    yolo_category_id = category_mapping[coco_category_id]
                    
                    # 获取分割多边形
                    segmentation = ann['segmentation']
                    # COCO格式的segmentation可能是多个多边形的列表
                    for polygon in segmentation:
                        # ✅ 1. 过滤空多边形和少于3个点的非法形状
                        if len(polygon) < 6:  # 3点 × 2坐标
                            print(f"警告: 多边形点不足3个，跳过 (长度: {len(polygon)})")
                            skipped_annotations += 1
                            continue
                        
                        # 确保多边形点的数量是偶数
                        if len(polygon) % 2 != 0:
                            print(f"警告: 多边形点数量不是偶数: {len(polygon)}")
                            skipped_annotations += 1
                            continue
                        
                        # 归一化坐标
                        normalized_points = []
                        for i in range(0, len(polygon), 2):
                            x = polygon[i] / image_width  # 归一化x坐标
                            y = polygon[i+1] / image_height  # 归一化y坐标
                            
                            # 确保坐标在有效范围内
                            x = max(0.0, min(1.0, x))
                            y = max(0.0, min(1.0, y))
                            
                            normalized_points.append(x)
                            normalized_points.append(y)
                        
                        # ✅ 2. 强制保留至少3个有效点
                        if len(normalized_points) < 6:
                            print(f"警告: 归一化后点不足3个，跳过")
                            skipped_annotations += 1
                            continue
                        
                        # ✅ 3. 写出前再校验一次坐标范围
                        if any(float(p) < 0 or float(p) > 1 for p in normalized_points):
                            print(f"警告: 归一化坐标超出[0,1]，跳过")
                            skipped_annotations += 1
                            continue
                        
                        # 写入YOLO格式的标注
                        line = [str(yolo_category_id)] + [f"{p:.6f}" for p in normalized_points]
                        f.write(' '.join(line) + '\n')
                        processed_annotations += 1
                        split_annotations += 1
        
        print(f"{split_name}集: 处理了 {split_annotations} 个标注")
        
        # 复制图片到对应目录
        if images_source_dir:
            images_copied = 0
            for image_id in image_ids:
                if image_id in image_id_to_info:
                    image_info = image_id_to_info[image_id]
                    image_filename = image_info['file_name']
                    
                    # 构建源图片路径
                    source_image_path = os.path.join(images_source_dir, image_filename)
                    
                    # 如果源图片存在，复制到目标目录
                    if os.path.exists(source_image_path):
                        dest_image_path = os.path.join(image_output_dirs[split_name], image_filename)
                        
                        # 使用shutil.copy2保留文件元数据
                        import shutil
                        shutil.copy2(source_image_path, dest_image_path)
                        images_copied += 1
                    else:
                        print(f"警告: 图片文件不存在: {source_image_path}")
            
            print(f"{split_name}集: 复制了 {images_copied} 张图片到 {image_output_dirs[split_name]}")
        
        # ✅ 4. 处理没有任何标注的图片，生成空标签文件
        processed_images = set()
        for image_id in image_ids:
            if image_id in image_id_to_info:
                image_info = image_id_to_info[image_id]
                image_filename = image_info['file_name']
                label_filename = os.path.splitext(image_filename)[0] + '.txt'
                label_path = os.path.join(output_dirs[split_name], label_filename)
                
                # 如果这个标签文件还不存在，说明该图片没有任何有效标注，创建空文件
                if not os.path.exists(label_path):
                    open(label_path, 'w').close()  # 创建空文件
    
    print(f"\n转换完成！")
    print(f"总标注数: {total_annotations}")
    print(f"成功处理: {processed_annotations}")
    print(f"跳过: {skipped_annotations}")
    print(f"输出目录: {output_base_dir}")
    print(f"train/val/labels 分别保存在: {list(output_dirs.values())}")
    
    # 创建YOLO数据配置文件
    create_yolo_data_config(output_base_dir, category_mapping)


def create_yolo_data_config(output_base_dir, category_mapping):
    """
    创建YOLO格式的数据配置文件data.yaml
    
    Args:
        output_base_dir: 输出基础目录
        category_mapping: 类别映射字典 {coco_id: yolo_id}
    """
    # 根据category_mapping反向获取类别名称
    # 这里假设类别ID 1,2,3对应ship,container,crane
    category_names = {
        0: 'ship',
        1: 'container', 
        2: 'crane'
    }
    
    # 构建YAML配置内容
    yaml_content = f"""# YOLO dataset configuration
path: {output_base_dir}  # dataset root dir
train: train/images  # train images (relative to 'path')
val: val/images  # val images (relative to 'path')
test: test/images  # test images (optional)

# Classes
nc: {len(category_names)}  # number of classes
names: {list(category_names.values())}  # class names
"""
    
    # 写入配置文件
    config_path = os.path.join(output_base_dir, 'data.yaml')
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"已创建YOLO数据配置文件: {config_path}")
    print(f"类别配置: {category_names}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将COCO格式的多边形标注转换为YOLOv8-seg格式并按train/val/test划分')
    parser.add_argument('--coco-json', default='harbor_port_backup/annotations.json', help='COCO格式的JSON文件路径')
    parser.add_argument('--output-base-dir', default='harbor_port_backup', help='输出基础目录')
    parser.add_argument('--images-source-dir', default=None, help='图片源目录，如果提供则会复制图片到对应目录')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='训练集比例')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='测试集比例')
    
    args = parser.parse_args()
    
    # 设置默认路径
    coco_json_path = os.path.join(os.getcwd(), args.coco_json)
    output_base_dir = os.path.join(os.getcwd(), args.output_base_dir)
    
    # 计算划分比例
    split_ratios = [args.train_ratio, args.val_ratio, args.test_ratio]
    
    print(f"开始转换...")
    print(f"COCO JSON文件: {coco_json_path}")
    print(f"输出基础目录: {output_base_dir}")
    print(f"数据集划分比例: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
    if args.images_source_dir:
        print(f"图片源目录: {args.images_source_dir}")
    
    convert_coco_to_yolo_seg_with_split(coco_json_path, output_base_dir, split_ratios, args.images_source_dir)