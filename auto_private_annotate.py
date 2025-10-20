import os
from pathlib import Path
import shutil

# 自制数据集标注
datasets = [
    ("datasets/private_ship/JPEGImages", "raw_private_ship", 0),      # ship
    ("datasets/private_container/JPEGImages", "raw_private_container", 1),  # container  
    ("datasets/private_crane/JPEGImages", "raw_private_crane", 2)     # crane
]

for jpeg_dir, output_dir, class_id in datasets:
    jpeg_path = Path(jpeg_dir)
    output_path = Path(output_dir)
    
    if not jpeg_path.exists():
        print(f"跳过 {jpeg_dir} - 目录不存在")
        continue
    
    # 创建输出目录
    (output_path / "images").mkdir(parents=True, exist_ok=True)
    (output_path / "labels").mkdir(parents=True, exist_ok=True)
    
    count = 0
    for img_path in jpeg_path.glob("*.jpg"):
        # 复制图片
        dst_img = output_path / "images" / img_path.name
        shutil.copy2(img_path, dst_img)
        
        # 生成全图 YOLO 标注
        label_path = output_path / "labels" / (img_path.stem + ".txt")
        with open(label_path, "w") as f:
            f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")  # 类别ID，中心点(0.5,0.5)，宽高(1.0,1.0)
        
        count += 1
    
    class_names = ["ship", "container", "crane"]
    print(f"{class_names[class_id]} 标注完成: {count} 张图片 (类别ID={class_id})")

print("所有自制数据集标注完成！")

