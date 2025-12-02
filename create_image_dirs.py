import json
import os
import shutil
from collections import defaultdict

def create_image_directory_structure():
    """创建图片目录结构并生成数据配置文件"""
    
    # 读取annotations.json获取图片信息
    with open('harbor_port_backup/annotations.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 获取所有图片文件名
    image_files = [img['file_name'] for img in data['images']]
    print(f"找到 {len(image_files)} 个图片文件")
    
    # 创建图片目录
    image_dirs = {
        'train': 'harbor_port_backup/train/images',
        'val': 'harbor_port_backup/val/images', 
        'test': 'harbor_port_backup/test/images'
    }
    
    for dir_path in image_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # 读取已有的标注文件来确定划分
    train_labels = os.listdir('harbor_port_backup/train/labels')
    val_labels = os.listdir('harbor_port_backup/val/labels')
    test_labels = os.listdir('harbor_port_backup/test/labels')
    
    # 提取图片文件名（不含扩展名）
    train_images = set([label.replace('.txt', '.jpg') for label in train_labels])
    val_images = set([label.replace('.txt', '.jpg') for label in val_labels])
    test_images = set([label.replace('.txt', '.jpg') for label in test_labels])
    
    print(f"训练集: {len(train_images)} 张图片")
    print(f"验证集: {len(val_images)} 张图片")
    print(f"测试集: {len(test_images)} 张图片")
    
    # 生成YOLO数据配置文件
    yaml_content = f"""# 港口数据集配置
train: train/images
val: val/images
test: test/images

nc: 3
names: ['ship', 'container', 'crane']

# 数据集统计信息
# 训练集: {len(train_images)} 张图片
# 验证集: {len(val_images)} 张图片  
# 测试集: {len(test_images)} 张图片
# 总类别: 3个 (ship, container, crane)
"""
    
    with open('harbor_port_backup/data.yaml', 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print("\n目录结构已创建:")
    print("harbor_port_backup/")
    print("├── train/")
    print("│   ├── images/  (需要放入图片)")
    print("│   └── labels/  (已生成)")
    print("├── val/")
    print("│   ├── images/  (需要放入图片)")
    print("│   └── labels/  (已生成)")
    print("├── test/")
    print("│   ├── images/  (需要放入图片)")
    print("│   └── labels/  (已生成)")
    print("└── data.yaml    (已生成)")
    
    print("\n下一步:")
    print("1. 将所有图片文件放入 harbor_port_backup/images/ 目录")
    print("2. 运行 organize_images.py 自动分配到 train/val/test/images/")

if __name__ == "__main__":
    create_image_directory_structure()