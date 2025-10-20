import argparse
import shutil
from pathlib import Path
from tqdm import tqdm

def merge_category_datasets(source_patterns, target_dir):
    """合并分类的数据集到统一目录"""
    # 创建目标目录结构
    target_images = target_dir / "images"
    target_labels = target_dir / "labels"
    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)
    
    total_files = 0
    
    # 处理每个分类的数据集
    for pattern, class_id in source_patterns:
        source_dir = Path(pattern)
        if not source_dir.exists():
            print(f"警告: 目录不存在，跳过: {source_dir}")
            continue
        
        # 获取所有图片文件
        image_files = list(source_dir.glob("**/*.jpg")) + list(source_dir.glob("**/*.png"))
        
        print(f"处理分类 {class_id}，找到 {len(image_files)} 张图片")
        
        # 复制文件并更新标签
        for img_path in tqdm(image_files, desc=f"处理分类 {class_id}"):
            # 生成目标文件名（避免冲突）
            base_name = f"{class_id}_{img_path.stem}{img_path.suffix}"
            target_img_path = target_images / base_name
            
            # 复制图片
            shutil.copy2(img_path, target_img_path)
            
            # 检查是否有对应的标签文件
            label_stem = img_path.stem
            for label_path in [
                img_path.parent.parent / "labels" / f"{label_stem}.txt",
                img_path.with_suffix(".txt")
            ]:
                if label_path.exists():
                    # 如果找到标签文件，复制它
                    target_label_path = target_labels / f"{class_id}_{label_stem}.txt"
                    shutil.copy2(label_path, target_label_path)
                    break
            else:
                # 如果没有找到标签文件，创建一个默认标签
                target_label_path = target_labels / f"{class_id}_{label_stem}.txt"
                with open(target_label_path, "w") as f:
                    f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
            
            total_files += 1
    
    print(f"合并完成，总共处理了 {total_files} 个文件")

def main():
    parser = argparse.ArgumentParser(description="合并分类数据集到统一目录")
    parser.add_argument("--public", action="store_true", help="合并公开数据集")
    parser.add_argument("--private", action="store_true", help="合并自制数据集")
    parser.add_argument("--all", action="store_true", help="合并所有数据集")
    args = parser.parse_args()
    
    if args.all or args.public:
        print("开始合并公开数据集...")
        public_patterns = [
            ("raw_public_ship/images", 0),     # ship
            ("raw_public_container/images", 1),  # container
            ("raw_public_crane/images", 2)      # crane
        ]
        merge_category_datasets(public_patterns, Path("raw_public"))
    
    if args.all or args.private:
        print("开始合并自制数据集...")
        private_patterns = [
            ("raw_private_ship/images", 0),     # ship
            ("raw_private_container/images", 1),  # container
            ("raw_private_crane/images", 2)      # crane
        ]
        merge_category_datasets(private_patterns, Path("raw_private"))
    
    if not any([args.all, args.public, args.private]):
        print("请选择要合并的数据集类型。使用 --help 查看可用选项。")

if __name__ == "__main__":
    main()