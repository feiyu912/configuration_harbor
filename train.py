import argparse
import subprocess
import sys
from pathlib import Path
import shutil

# 数据集配置路径
CONFIGS_DIR = Path("configs")
PUBLIC_CONFIG = CONFIGS_DIR / "public_dataset.yaml"
PRIVATE_CONFIG = CONFIGS_DIR / "private_dataset.yaml"
MIXED_CONFIG = CONFIGS_DIR / "mixed_dataset.yaml"

def run_command(cmd, check=True):
    """运行命令并返回结果"""
    print(f"执行命令: {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)

def run_data_processor(mode, dataset_type=None, normalize=False, denoise=None, augment=False, aug_per_image=2):
    """运行数据处理模块"""
    cmd = [
        sys.executable,
        str(Path('enhanced_data_processor.py')),
        mode
    ]
    
    if mode == "merge":
        if dataset_type == "public":
            cmd.append("--public")
        elif dataset_type == "private":
            cmd.append("--private")
        elif dataset_type == "all":
            cmd.append("--all")
    elif mode == "process":
        if dataset_type == "public":
            cmd.extend(["--raw-dir", "raw_public", "--out-dir", "dataset_yolo_public"])
        elif dataset_type == "private":
            cmd.extend(["--raw-dir", "raw_private", "--out-dir", "dataset_yolo_private"])
        elif dataset_type == "mixed":
            cmd.extend(["--raw-dir", "raw_mixed", "--out-dir", "dataset_yolo"])
    
    # 添加数据增强选项
    if normalize:
        cmd.append("--normalize")
    if denoise:
        cmd.extend(["--denoise", denoise])
    if augment and mode == "process":
        cmd.append("--augment")
        cmd.extend(["--aug-per-image", str(aug_per_image)])
    
    run_command(cmd)

def run_train(data_config, name, compare_config=None, model='yolov8n.pt', epochs=50, batch=16):
    """运行训练脚本"""
    cmd = [
        sys.executable,
        str(Path("src") / "train.py"),
        "--data", str(data_config),
        "--name", name,
        "--model", model,
        "--epochs", str(epochs)
    ]
    
    if compare_config:
        cmd.extend(["--compare-data", str(compare_config)])
    
    print(f"执行命令: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="港口目标检测模型训练入口脚本")
    
    # 数据集选择组
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument("--public", action="store_true", help="仅使用公开数据集训练")
    dataset_group.add_argument("--private", action="store_true", help="仅使用自制数据集训练")
    dataset_group.add_argument("--mixed", action="store_true", help="使用混合数据集训练")
    dataset_group.add_argument("--compare", action="store_true", help="比较公开和自制数据集的训练结果")
    
    # 训练参数组
    parser.add_argument("--model", default="yolov8n.pt", help="预训练模型路径")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch", type=int, default=16, help="批次大小")
    
    # 数据处理增强选项
    parser.add_argument('--normalize', action='store_true', help="对图像进行归一化处理")
    parser.add_argument('--denoise', choices=['gaussian', 'median', 'bilateral'], help="图像去噪方法")
    parser.add_argument('--augment', action='store_true', help="启用数据增强")
    parser.add_argument('--aug-per-image', type=int, default=2, help="每个图像生成的增强样本数")
    parser.add_argument('--skip-preprocess', action='store_true', help="跳过数据预处理步骤")
    
    args = parser.parse_args()
    
    # 默认选择
    if not any([args.public, args.private, args.mixed, args.compare]):
        print("请选择训练数据集类型。使用 --help 查看可用选项。")
        parser.print_help()
        return
    
    # 运行对应的训练流程
    if args.public:
        print("===== 开始处理和训练公开数据集 =====")
        if not args.skip_preprocess:
            print("1. 合并公开数据集...")
            run_data_processor("merge", "public", args.normalize, args.denoise)
            print("2. 处理和分割公开数据集...")
            run_data_processor("process", "public", args.normalize, args.denoise, args.augment, args.aug_per_image)
        print("3. 训练公开数据集...")
        run_train(PUBLIC_CONFIG, "port_public", model=args.model, epochs=args.epochs, batch=args.batch)
    
    elif args.private:
        print("===== 开始处理和训练自制数据集 =====")
        if not args.skip_preprocess:
            print("1. 合并自制数据集...")
            run_data_processor("merge", "private", args.normalize, args.denoise)
            print("2. 处理和分割自制数据集...")
            run_data_processor("process", "private", args.normalize, args.denoise, args.augment, args.aug_per_image)
        print("3. 训练自制数据集...")
        run_train(PRIVATE_CONFIG, "port_private", model=args.model, epochs=args.epochs, batch=args.batch)
    
    elif args.mixed:
        print("===== 开始处理和训练混合数据集 =====")
        if not args.skip_preprocess:
            print("1. 合并所有数据集...")
            run_data_processor("merge", "all", args.normalize, args.denoise)
            
            # 创建混合数据集目录
            raw_mixed = Path("raw_mixed")
            raw_mixed.mkdir(exist_ok=True)
            (raw_mixed / "images").mkdir(exist_ok=True)
            (raw_mixed / "labels").mkdir(exist_ok=True)
            
            # 复制公开数据集到混合数据集
            public_images = list(Path("raw_public/images").glob("*.*"))
            for img in public_images:
                shutil.copy2(img, raw_mixed / "images" / f"public_{img.name}")
                label_path = Path("raw_public/labels") / f"{img.stem}.txt"
                if label_path.exists():
                    shutil.copy2(label_path, raw_mixed / "labels" / f"public_{img.stem}.txt")
            
            # 复制自制数据集到混合数据集
            private_images = list(Path("raw_private/images").glob("*.*"))
            for img in private_images:
                shutil.copy2(img, raw_mixed / "images" / f"private_{img.name}")
                label_path = Path("raw_private/labels") / f"{img.stem}.txt"
                if label_path.exists():
                    shutil.copy2(label_path, raw_mixed / "labels" / f"private_{img.stem}.txt")
            
            print("2. 处理和分割混合数据集...")
            run_data_processor("process", "mixed", args.normalize, args.denoise, args.augment, args.aug_per_image)
        print("3. 训练混合数据集...")
        run_train(MIXED_CONFIG, "port_mixed", model=args.model, epochs=args.epochs, batch=args.batch)
    
    elif args.compare:
        print("===== 比较公开和自制数据集的训练结果 =====")
        if not args.skip_preprocess:
            # 处理两个数据集
            print("处理公开数据集...")
            run_data_processor("merge", "public", args.normalize, args.denoise)
            run_data_processor("process", "public", args.normalize, args.denoise, args.augment, args.aug_per_image)
            
            print("处理自制数据集...")
            run_data_processor("merge", "private", args.normalize, args.denoise)
            run_data_processor("process", "private", args.normalize, args.denoise, args.augment, args.aug_per_image)
        
        # 先训练公开数据集作为主数据集
        run_train(PUBLIC_CONFIG, "port_public", PRIVATE_CONFIG, model=args.model, epochs=args.epochs, batch=args.batch)
        # 再训练自制数据集作为主数据集
        run_train(PRIVATE_CONFIG, "port_private", PUBLIC_CONFIG, model=args.model, epochs=args.epochs, batch=args.batch)

if __name__ == "__main__":
    main()