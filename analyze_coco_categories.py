import json
from pathlib import Path

# COCO JSON文件路径
coco_json_paths = [
    'g:\\configuration_harbor\\labels_my-project-name_2025-11-07-11-34-22.json'
]

print("分析COCO JSON文件中的类别定义...\n")

# 遍历所有COCO JSON文件
for json_path in coco_json_paths:
    if not Path(json_path).exists():
        print(f"警告: 文件不存在 - {json_path}")
        continue
    
    print(f"文件: {Path(json_path).name}")
    print("="*50)
    
    # 读取COCO JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 检查是否有categories字段
    if 'categories' in data:
        print("类别定义:")
        for category in data['categories']:
            cat_id = category.get('id')
            cat_name = category.get('name', 'Unknown')
            print(f"  - ID: {cat_id}, 名称: {cat_name}")
        print()
    else:
        print("未找到categories字段\n")
    
    # 分析annotations中的category_id分布
    if 'annotations' in data:
        category_counts = {}
        for ann in data['annotations']:
            cat_id = ann.get('category_id')
            category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
        
        print("标注中使用的类别ID统计:")
        for cat_id, count in sorted(category_counts.items()):
            print(f"  - 类别ID {cat_id}: {count} 个标注")
        print()
    else:
        print("未找到annotations字段\n")

print("分析完成！")
print("\n请根据上述分析结果，修改coco_to_yolo_converter.py中的category_map字典，确保正确映射到0:ship, 1:container, 2:crane")