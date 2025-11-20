import argparse
import random
import shutil
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm
import os

# 设置随机种子确保可重现性
random.seed(42)
np.random.seed(42)

def copy_with_dirs(src: Path, dst: Path) -> None:
    """复制文件并确保目标目录存在"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def read_yolo_labels(label_path: Path):
    """读取YOLO格式标签"""
    if not label_path.exists():
        return []
    items = []
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    try:
                        cls = int(parts[0])
                        x, y, w, h = map(float, parts[1:])
                        # 验证标签有效性
                        if 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1:
                            items.append((cls, x, y, w, h))
                    except ValueError:
                        continue
    except Exception as e:
        print(f"读取标签文件错误 {label_path}: {e}")
    return items

def write_yolo_labels(label_path: Path, labels) -> None:
    """写入YOLO格式标签"""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w", encoding="utf-8") as f:
        for cls, x, y, w, h in labels:
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def normalize_image(image):
    """图像归一化处理"""
    try:
        # 转换为浮点型
        img_float = image.astype(np.float32)
        
        # 计算均值和标准差，避免除以零
        mean_val = np.mean(img_float)
        std_val = np.std(img_float)
        
        # 全局对比度归一化
        if std_val > 0:
            img_float = (img_float - mean_val) / std_val
        
        # 缩放到0-255范围
        img_normalized = cv2.normalize(img_float, None, 0, 255, cv2.NORM_MINMAX)
        return img_normalized.astype(np.uint8)
    except Exception as e:
        print(f"归一化图像时出错: {e}")
        # 返回原始图像的副本
        return image.copy()

def denoise_image(image, method='gaussian'):
    """图像去噪处理"""
    if method == 'gaussian':
        # 高斯滤波去噪
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif method == 'median':
        # 中值滤波去噪
        return cv2.medianBlur(image, 5)
    elif method == 'bilateral':
        # 双边滤波去噪（保留边缘）
        return cv2.bilateralFilter(image, 9, 75, 75)
    else:
        return image

def handle_missing_labels(image_path, label_path, class_id=None):
    """处理缺失标签的情况"""
    if not label_path.exists():
        # 根据文件名模式或默认类别创建标签
        if class_id is not None:
            # 创建默认标签（整个图像为目标）
            with open(label_path, "w") as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
            return True
        return False
    return True

def validate_and_fix_labels(labels, image_shape):
    """验证和修复YOLO格式标签，确保坐标在0-1范围内"""
    valid_labels = []
    h, w = image_shape[:2]
    
    for label in labels:
        try:
            cls, x, y, bw, bh = label
            
            # 确保坐标在有效范围内
            x = max(0, min(1, x))
            y = max(0, min(1, y))
            bw = max(0.01, min(1, bw))
            bh = max(0.01, min(1, bh))
            
            # 确保边界框在图像内
            half_bw = bw / 2
            half_bh = bh / 2
            if x - half_bw < 0:
                x = half_bw
            if x + half_bw > 1:
                x = 1 - half_bw
            if y - half_bh < 0:
                y = half_bh
            if y + half_bh > 1:
                y = 1 - half_bh
            
            valid_labels.append((cls, x, y, bw, bh))
        except Exception as e:
            print(f"修复标签时出错: {e}")
            continue
    
    return valid_labels

def horizontal_flip(image, labels):
    """水平翻转增强"""
    h, w = image.shape[:2]
    flipped = cv2.flip(image, 1)
    new_labels = []
    for cls, x, y, bw, bh in labels:
        new_labels.append((cls, 1.0 - x, y, bw, bh))
    return flipped, new_labels

def rotate_image(image, angle):
    """图像旋转"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return rotated, M

def rotate_labels(labels, angle, img_h, img_w):
    """旋转标签坐标"""
    center = (img_w / 2, img_h / 2)
    angle_rad = np.radians(angle)
    new_labels = []
    
    for cls, x, y, bw, bh in labels:
        # 转换为图像坐标（非归一化）
        img_x = x * img_w
        img_y = y * img_h
        
        # 计算相对于中心的坐标
        rel_x = img_x - center[0]
        rel_y = img_y - center[1]
        
        # 旋转坐标
        new_rel_x = rel_x * np.cos(angle_rad) - rel_y * np.sin(angle_rad)
        new_rel_y = rel_x * np.sin(angle_rad) + rel_y * np.cos(angle_rad)
        
        # 转换回图像坐标
        new_img_x = new_rel_x + center[0]
        new_img_y = new_rel_y + center[1]
        
        # 归一化
        new_x = new_img_x / img_w
        new_y = new_img_y / img_h
        
        # 确保坐标在有效范围内
        if 0 <= new_x <= 1 and 0 <= new_y <= 1:
            new_labels.append((cls, new_x, new_y, bw, bh))
    
    return new_labels

def adjust_brightness_contrast(image, brightness_factor, contrast_factor):
    """调整亮度和对比度"""
    adjusted = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=brightness_factor)
    return adjusted

def augment_image(image, labels, aug_type):
    """多种数据增强方法"""
    h, w = image.shape[:2]
    
    if aug_type == 'flip':
        return horizontal_flip(image, labels)
    elif aug_type == 'rotate':
        angle = np.random.uniform(-15, 15)  # 随机旋转角度
        rotated_img, M = rotate_image(image, angle)
        rotated_labels = rotate_labels(labels, angle, h, w)
        return rotated_img, rotated_labels
    elif aug_type == 'brightness':
        # 随机调整亮度
        brightness = np.random.uniform(-20, 20)
        contrast = np.random.uniform(0.8, 1.2)
        bright_img = adjust_brightness_contrast(image, brightness, contrast)
        return bright_img, labels.copy()
    elif aug_type == 'noise':
        # 添加少量高斯噪声
        noise = np.random.normal(0, 5, image.shape).astype(np.uint8)
        noisy_img = cv2.add(image, noise)
        return noisy_img, labels.copy()
    else:
        return image, labels.copy()

def validate_and_fix_labels(labels, img_shape):
    """验证并修复标签，确保它们在有效范围内"""
    valid_labels = []
    h, w = img_shape[:2]
    
    for cls, x, y, bw, bh in labels:
        # 确保坐标和尺寸在有效范围内
        x = max(0, min(1, x))
        y = max(0, min(1, y))
        bw = max(0.01, min(1, bw))
        bh = max(0.01, min(1, bh))
        
        # 确保边界框不超出图像
        half_bw = bw / 2
        half_bh = bh / 2
        x = max(half_bw, min(1 - half_bw, x))
        y = max(half_bh, min(1 - half_bh, y))
        
        valid_labels.append((cls, x, y, bw, bh))
    
    return valid_labels

def process_image(image_path, label_path, out_image_path, out_label_path, 
                 normalize=False, denoise=None, fix_labels=True, class_id=None):
    """处理单个图像和标签文件"""
    # 读取图像
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"警告: 无法读取图像 {image_path}")
        return False
    
    # 读取标签
    labels = read_yolo_labels(label_path)
    
    # 处理缺失标签
    if not labels and class_id is not None:
        handle_missing_labels(image_path, label_path, class_id)
        labels = read_yolo_labels(label_path)
    
    # 图像预处理
    if normalize:
        image = normalize_image(image)
    
    if denoise:
        image = denoise_image(image, denoise)
    
    # 验证和修复标签
    if fix_labels:
        labels = validate_and_fix_labels(labels, image.shape)
    
    # 保存处理后的图像和标签
    cv2.imwrite(str(out_image_path), image)
    write_yolo_labels(out_label_path, labels)
    
    return True

def split_and_process_dataset(raw_dir: Path, out_dir: Path, val_ratio: float, test_ratio: float,
                             normalize=False, denoise=None, fix_labels=True, class_map=None):
    """分割数据集并处理图像"""
    img_dir = raw_dir / "images"
    lbl_dir = raw_dir / "labels"
    
    # 获取所有图像文件
    images = sorted([p for p in img_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    random.shuffle(images)
    
    n = len(images)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    
    splits = {
        "train": images[: n - n_val - n_test],
        "val": images[n - n_val - n_test : n - n_test],
        "test": images[n - n_test :],
    }
    
    total_processed = 0
    for split_name, split_images in splits.items():
        for img_path in tqdm(split_images, desc=f"处理 {split_name} 集"):
            rel = img_path.relative_to(img_dir)
            dst_img = out_dir / "images" / split_name / rel
            
            # 确定标签路径
            lbl_rel = rel.with_suffix(".txt")
            src_lbl = lbl_dir / lbl_rel
            dst_lbl = out_dir / "labels" / split_name / lbl_rel
            
            # 尝试从文件名获取类别信息（如果提供了class_map）
            class_id = None
            if class_map:
                for cls_name, cls_idx in class_map.items():
                    if cls_name.lower() in img_path.name.lower():
                        class_id = cls_idx
                        break
            
            # 处理图像和标签
            if process_image(img_path, src_lbl, dst_img, dst_lbl, normalize, denoise, fix_labels, class_id):
                total_processed += 1
    
    print(f"数据处理完成，成功处理 {total_processed} 个文件")

def enhanced_augmentation(out_dir: Path, aug_per_image: int):
    """增强的数据增强功能"""
    train_images = list((out_dir / "images" / "train").rglob("*.jpg")) + list(
        (out_dir / "images" / "train").rglob("*.png")
    )
    
    # 增强方法列表
    aug_methods = ['flip', 'rotate', 'brightness', 'noise']
    
    for img_path in tqdm(train_images, desc="增强数据"):
        label_path = out_dir / "labels" / "train" / img_path.relative_to(
            out_dir / "images" / "train"
        ).with_suffix(".txt")
        
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        labels = read_yolo_labels(label_path)
        if not labels:
            continue
        
        # 为每个图像生成多个增强样本
        for k in range(aug_per_image):
            # 随机选择增强方法
            aug_type = random.choice(aug_methods)
            aug_img, aug_labels = augment_image(image, labels, aug_type)
            
            # 保存增强后的图像和标签
            aug_name = f"{img_path.stem}_aug{aug_type}{k}"
            dst_img = img_path.with_name(aug_name + img_path.suffix)
            dst_lbl = label_path.with_name(aug_name + ".txt")
            
            cv2.imwrite(str(dst_img), aug_img)
            write_yolo_labels(dst_lbl, aug_labels)
    
    print(f"数据增强完成，为每个图像生成了 {aug_per_image} 个增强样本")

def merge_categorized_datasets(source_patterns, target_dir, normalize=False, denoise=None):
    """合并分类数据集并进行预处理"""
    # 创建目标目录结构
    target_images = target_dir / "images"
    target_labels = target_dir / "labels"
    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)
    
    total_files = 0
    success_count = 0
    failed_files = 0
    
    # 处理每个分类的数据集
    for pattern, class_id in source_patterns:
        source_dir = Path(pattern)
        if not source_dir.exists():
            print(f"警告: 目录不存在，跳过: {source_dir}")
            continue
        
        try:
            # 获取所有图片文件 - 更好地处理中文文件名
            image_files = []
            # 使用os.listdir替代glob以更好地处理中文路径
            if source_dir.exists():
                for root, _, files in os.walk(source_dir):
                    for file in files:
                        ext = os.path.splitext(file)[1].lower()
                        if ext in ['.jpg', '.jpeg', '.png']:
                            img_path = Path(root) / file
                            image_files.append(img_path)
            
            print(f"处理分类 {class_id}，找到 {len(image_files)} 张图片")
            
            # 复制文件并处理
            for img_path in tqdm(image_files, desc=f"处理分类 {class_id}"):
                total_files += 1
                
                try:
                    # 生成安全的目标文件名（避免编码问题）
                    safe_name = f"{class_id}_{total_files:06d}{img_path.suffix}"
                    target_img_path = target_images / safe_name
                    target_label_path = target_labels / f"{class_id}_{total_files:06d}.txt"
                    
                    # 读取图像 - 增强对中文文件名的支持
                    print(f"尝试读取图像: {img_path}")
                    image = cv2.imread(str(img_path))
                    
                    # 如果OpenCV读取失败，尝试使用PIL作为备选
                    if image is None:
                        print(f"警告: OpenCV无法读取图像 {img_path}，尝试使用PIL")
                        try:
                            from PIL import Image
                            import numpy as np
                            # 使用PIL打开图像
                            pil_img = Image.open(str(img_path))
                            # 转换为OpenCV格式
                            image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                            print(f"成功使用PIL读取图像")
                        except Exception as pil_e:
                            print(f"PIL也无法读取图像: {pil_e}")
                            failed_files += 1
                            continue
                    
                    # 图像预处理
                    if normalize:
                        image = normalize_image(image)
                    if denoise:
                        image = denoise_image(image, denoise)
                    
                    # 保存处理后的图像
                    success = cv2.imwrite(str(target_img_path), image)
                    if not success:
                        print(f"警告: 无法保存图像 {target_img_path}")
                        failed_files += 1
                        continue
                    
                    # 处理标签
                    label_stem = img_path.stem
                    found_label = False
                    
                    # 检查可能的标签文件位置
                    for label_path in [
                        img_path.parent.parent / "labels" / f"{label_stem}.txt",
                        img_path.with_suffix(".txt")
                    ]:
                        if label_path.exists():
                            try:
                                # 读取并处理标签
                                labels = read_yolo_labels(label_path)
                                
                                # 验证和修复标签
                                valid_labels = validate_and_fix_labels(labels, image.shape)
                                write_yolo_labels(target_label_path, valid_labels)
                                found_label = True
                            except Exception as e:
                                print(f"处理标签文件时出错 {label_path}: {e}")
                            break
                    
                    if not found_label:
                        # 创建默认标签
                        try:
                            with open(target_label_path, "w") as f:
                                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
                        except Exception as e:
                            print(f"创建默认标签时出错 {target_label_path}: {e}")
                            continue
                    
                    success_count += 1
                except Exception as e:
                    print(f"处理文件时出错 {img_path}: {e}")
                    failed_files += 1
                    # 继续处理下一个文件
                    continue
        except Exception as e:
            print(f"处理分类 {class_id} 时出错: {e}")
            continue
    
    print(f"合并和预处理完成，总共处理了 {total_files} 个文件，成功 {success_count} 个，失败 {failed_files} 个")

def main():
    parser = argparse.ArgumentParser(description="增强版YOLO数据集处理工具")
    
    # 子命令解析器
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 合并分类数据集命令
    merge_parser = subparsers.add_parser("merge", help="合并分类数据集")
    merge_parser.add_argument("--public", action="store_true", help="合并公开数据集")
    merge_parser.add_argument("--private", action="store_true", help="合并自制数据集")
    merge_parser.add_argument("--all", action="store_true", help="合并所有数据集")
    merge_parser.add_argument("--normalize", action="store_true", help="对图像进行归一化")
    merge_parser.add_argument("--denoise", choices=["gaussian", "median", "bilateral"], help="图像去噪方法")
    
    # 处理和分割数据集命令
    process_parser = subparsers.add_parser("process", help="处理和分割数据集")
    process_parser.add_argument("--raw-dir", type=Path, required=True, help="原始数据集目录")
    process_parser.add_argument("--out-dir", type=Path, required=True, help="输出目录")
    process_parser.add_argument("--val-ratio", type=float, default=0.15, help="验证集比例")
    process_parser.add_argument("--test-ratio", type=float, default=0.15, help="测试集比例")
    process_parser.add_argument("--normalize", action="store_true", help="对图像进行归一化")
    process_parser.add_argument("--denoise", choices=["gaussian", "median", "bilateral"], help="图像去噪方法")
    process_parser.add_argument("--fix-labels", action="store_true", default=True, help="验证和修复标签")
    process_parser.add_argument("--augment", action="store_true", help="启用数据增强")
    process_parser.add_argument("--aug-per-image", type=int, default=2, help="每个图像生成的增强样本数")
    
    args = parser.parse_args()
    
    # 类别映射
    class_map = {
        "ship": 0,
        "container": 1,
        "crane": 2
    }
    
    if args.command == "merge":
        # 合并数据集命令
        if args.all or args.public:
            print("开始合并公开数据集...")
            public_patterns = [
                ("raw_public_ship/images", 0),     # ship
                ("raw_public_container/images", 1),  # container
                ("raw_public_crane/images", 2)      # crane
            ]
            merge_categorized_datasets(public_patterns, Path("raw_public"), args.normalize, args.denoise)
        
        if args.all or args.private:
            print("开始合并自制数据集...")
            private_patterns = [
                ("raw_private_ship/images", 0),     # ship
                ("raw_private_container/images", 1),  # container
                ("raw_private_crane/images", 2)      # crane
            ]
            merge_categorized_datasets(private_patterns, Path("raw_private"), args.normalize, args.denoise)
        
        if not any([args.all, args.public, args.private]):
            print("请选择要合并的数据集类型。使用 --help 查看可用选项。")
    
    elif args.command == "process":
        # 处理和分割数据集命令
        print(f"开始处理数据集，从 {args.raw_dir} 到 {args.out_dir}")
        
        # 确保输出目录存在
        args.out_dir.mkdir(parents=True, exist_ok=True)
        
        # 分割和处理数据集
        split_and_process_dataset(args.raw_dir, args.out_dir, args.val_ratio, args.test_ratio,
                                 args.normalize, args.denoise, args.fix_labels, class_map)
        
        # 数据增强
        if args.augment and args.aug_per_image > 0:
            enhanced_augmentation(args.out_dir, args.aug_per_image)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()