import json
import os
from pathlib import Path

def coco_to_yolo(coco_json_path, output_dir):
    """将COCO格式的JSON标注文件转换为YOLO格式的TXT文件
    
    Args:
        coco_json_path: COCO格式JSON文件路径
        output_dir: 输出YOLO格式TXT文件的目录
    """
    # 确保输出目录存在
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取COCO JSON文件
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 创建图像ID到文件名和尺寸的映射
    image_info = {}
    for img in data['images']:
        image_info[img['id']] = {
            'file_name': img['file_name'],
            'width': img['width'],
            'height': img['height']
        }
    
    # 按图像ID分组标注
    annotations_by_image = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    # 转换并保存YOLO格式标签
    processed_count = 0
    for img_id, annotations in annotations_by_image.items():
        if img_id not in image_info:
            print(f"警告: 找不到图像ID {img_id} 的信息，跳过")
            continue
        
        img_info = image_info[img_id]
        img_width = img_info['width']
        img_height = img_info['height']
        
        # 获取输出文件名（与图像同名但扩展名为txt）
        img_filename = Path(img_info['file_name']).stem
        output_txt_path = output_dir / f"{img_filename}.txt"
        
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            for ann in annotations:
                # 获取边界框
                bbox = ann['bbox']  # [x_min, y_min, width, height]
                
                # 计算YOLO格式的中心点坐标和宽高（归一化到0-1范围）
                x_center = (bbox[0] + bbox[2] / 2) / img_width
                y_center = (bbox[1] + bbox[3] / 2) / img_height
                width = bbox[2] / img_width
                height = bbox[3] / img_height
                
                # JSON类别ID映射：3->0, 1->1, 2->2
                # 最终txt中：0-ship, 1-container, 2-crane
                coco_category_id = ann.get('category_id', 0)
                category_map = {
                    3: 0,  # JSON中的3(ship) -> txt中的0(ship)
                    1: 1,  # JSON中的1(container) -> txt中的1(container)
                    2: 2   # JSON中的2(crane) -> txt中的2(crane)
                }
                class_id = category_map.get(coco_category_id, 0)
                
                # 写入YOLO格式行
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        print(f"已转换 {output_txt_path}")
        processed_count += 1
    
    print(f"从 {Path(coco_json_path).name} 转换完成！共处理 {processed_count} 张图像")
    return processed_count


def main():
    # 要处理的COCO JSON文件路径列表 - 用户提供的container标注文件
    coco_json_paths = [
        'g:\\configuration_harbor\\labels_my-project-name_2025-11-07-11-34-22.json'
    ]
    
    # 输出目录（raw_private_container/labels）
    output_dir = Path('g:\\configuration_harbor\\raw_private_container\labels')
    
    print(f"开始处理COCO JSON文件并转换到 {output_dir} 目录...")
    
    total_processed = 0
    for json_path in coco_json_paths:
        if Path(json_path).exists():
            print(f"\n处理文件: {json_path}")
            processed = coco_to_yolo(json_path, output_dir)
            total_processed += processed
        else:
            print(f"\n警告: 文件不存在 - {json_path}")
    
    print("\n===== 转换总结 =====")
    print(f"总共处理了 {total_processed} 张图像的标注数据")
    print(f"所有YOLO格式标签已保存到: {output_dir}")
    print("\nYOLO格式说明：")
    print("- 格式为每行一个对象：[class_id] [x_center] [y_center] [width] [height]")
    print("- 类别ID已按照统一标准映射：ship->0, container->1, crane->2")
    print("- COCO数据中的类别ID 3->ship(0), 1->container(1), 2->crane(2)")
    print("- 坐标已归一化到0-1范围")
    print("- 已替换raw_private_container/labels目录中的原有标签文件")


if __name__ == "__main__":
    main()