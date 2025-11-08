import argparse
import subprocess
import sys
import os
from pathlib import Path
import shutil

# 数据集配置路径
CONFIGS_DIR = Path("configs")
PUBLIC_CONFIG = CONFIGS_DIR / "public_dataset.yaml"
PRIVATE_CONFIG = CONFIGS_DIR / "private_dataset.yaml"
# 使用相对路径配置，避免路径错误


def run_command(cmd, check=True):
    """运行命令并返回结果"""
    print(f"执行命令: {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)

def run_preprocess_command(args):
    """运行预处理命令"""
    print("运行数据预处理...")
    
    # 根据compare模式选择不同的处理流程
    if args.compare:
        # 直接从分类原始数据开始处理
        print("使用分阶段比较模式，直接从分类原始数据开始...")
        
        # 构建预处理命令
        cmd = [
            "python", "enhanced_data_processor.py",
            "process_categorized",  # 新增命令，直接处理分类数据
            "--raw_public_ship", "raw_public_ship",
            "--raw_public_container", "raw_public_container", 
            "--raw_private_ship", "raw_private_ship",
            "--raw_private_container", "raw_private_container",
            "--raw_private_crane", "raw_private_crane",
            "--output_public", "dataset_yolo_public",
            "--output_private", "dataset_yolo_private",
            "--output_mixed", "dataset_yolo_mixed_test"
        ]
        
        # 添加预处理选项
        if args.normalize:
            cmd.append("--normalize")
        if args.augment:
            cmd.append("--augment")
        if args.denoise:
            cmd.append("--denoise")
        
        try:
            subprocess.run(cmd, check=True)
            print("✅ 分阶段数据预处理完成")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ 分阶段数据预处理失败: {e}")
            return False
    
    else:
        # 原始流程，使用合并后的数据
        cmd = [
            "python", "enhanced_data_processor.py",
            "process",
            "--raw_public", "raw_public",
            "--raw_private", "raw_private",
            "--output_public", "dataset_yolo_public",
            "--output_private", "dataset_yolo_private",
            "--output_mixed", "dataset_yolo_mixed_test"
        ]
        
        # 添加预处理选项
        if args.normalize:
            cmd.append("--normalize")
        if args.augment:
            cmd.append("--augment")
        if args.denoise:
            cmd.append("--denoise")
        
        try:
            subprocess.run(cmd, check=True)
            print("✅ 数据预处理完成")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ 数据预处理失败: {e}")
            return False

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
            cmd.extend(["--dataset-type", "public"])  # 添加数据集类型参数
        elif dataset_type == "private":
            cmd.extend(["--raw-dir", "raw_private", "--out-dir", "dataset_yolo_private"])
            cmd.extend(["--dataset-type", "private"])  # 添加数据集类型参数
    elif mode == "process_categorized":
        # 直接从分类原始数据开始处理，跳过中间合并步骤
        cmd.extend([
            "--raw_public_ship", "raw_public_ship",
            "--raw_public_container", "raw_public_container", 
            "--raw_private_ship", "raw_private_ship",
            "--raw_private_container", "raw_private_container",
            "--raw_private_crane", "raw_private_crane",
            "--output_public", "dataset_yolo_public",
            "--output_private", "dataset_yolo_private",
            "--output_mixed", "dataset_yolo_mixed_test"
        ])
    
    # 添加数据增强选项
    if normalize:
        cmd.append("--normalize")
    if denoise:
        cmd.extend(["--denoise", denoise])
    if augment and (mode == "process" or mode == "process_categorized"):
        cmd.append("--augment")
        cmd.extend(["--aug-per-image", str(aug_per_image)])
    
    run_command(cmd)

def run_train(data_config, name, model='yolov8n.pt', epochs=50, batch=16):
    """运行训练脚本"""
    cmd = [
        sys.executable,
        str(Path("src") / "train.py"),
        "--data", str(data_config),
        "--name", name,
        "--model", model,
        "--epochs", str(epochs)
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    run_command(cmd)

def run_eval_command(model_path, data_config, eval_name, model='yolov8n.pt'):
    """运行评估命令"""
    cmd = [
        sys.executable,
        str(Path("src") / "train.py"),
        "--data", str(data_config),
        "--name", f"eval_{eval_name}",
        "--model", str(model_path),
        "--epochs", "0"  # 0个epoch表示只评估不训练
    ]
    
    print(f"执行评估命令: {' '.join(cmd)}")
    result = run_command(cmd)
    return result

def save_compare_results(results_dir, public_results, private_results):
    """保存比较结果到文件"""
    import json
    from datetime import datetime
    
    os.makedirs(results_dir, exist_ok=True)
    
    result_file = Path(results_dir) / f"compare_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    compare_data = {
        "timestamp": datetime.now().isoformat(),
        "public_model": str(public_results.get('model_path', '')),
        "private_model": str(private_results.get('model_path', '')),
        "test_dataset": "dataset_yolo_mixed_test",
        "results": {
            "public_model_on_mixed": public_results,
            "private_model_on_mixed": private_results
        }
    }
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(compare_data, f, indent=2, ensure_ascii=False)
    
    print(f"比较结果已保存到: {result_file}")
    return result_file

def main():
    parser = argparse.ArgumentParser(description="港口目标检测模型训练入口脚本")
    
    # 数据集选择组
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument("--public", action="store_true", help="仅使用公开数据集训练")
    dataset_group.add_argument("--private", action="store_true", help="仅使用自制数据集训练")
    
    dataset_group.add_argument("--compare", action="store_true", help="比较公开和自制数据集的训练结果")
    
    # 分阶段执行选项
    parser.add_argument("--stage", choices=['preprocess', 'train', 'evaluate', 'all'], 
                       default='all', help="执行阶段：preprocess-仅预处理, train-仅训练, evaluate-仅评估, all-全流程")
    parser.add_argument("--compare-dir", type=str, default="compare_results", 
                       help="比较结果保存目录")
    
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
    if not any([args.public, args.private, args.compare]):
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
    

    
    elif args.compare:
        print("===== 比较公开和自制数据集的训练结果 =====")
        
        # 根据阶段执行相应流程
        if args.stage in ['preprocess', 'all']:
            print("\n--- 阶段1: 数据预处理 ---")
            # 处理两个数据集
            print("处理公开数据集...")
            run_data_processor("merge", "public", args.normalize, args.denoise)
            run_data_processor("process", "public", args.normalize, args.denoise, args.augment, args.aug_per_image)
            
            print("处理自制数据集...")
            run_data_processor("merge", "private", args.normalize, args.denoise)
            run_data_processor("process", "private", args.normalize, args.denoise, args.augment, args.aug_per_image)
            
            # 创建统一的混合测试集
            print("创建统一的混合测试集...")
            try:
                os.makedirs("dataset_yolo_mixed_test/images/test", exist_ok=True)
                os.makedirs("dataset_yolo_mixed_test/labels/test", exist_ok=True)
                
                test_images_count = 0
                test_labels_count = 0
                
                # 复制公开测试集
                if os.path.exists("dataset_yolo_public/images/test"):
                    for file in Path("dataset_yolo_public/images/test").glob("*"):
                        shutil.copy(file, Path("dataset_yolo_mixed_test/images/test") / file.name)
                        test_images_count += 1
                if os.path.exists("dataset_yolo_public/labels/test"):
                    for file in Path("dataset_yolo_public/labels/test").glob("*"):
                        shutil.copy(file, Path("dataset_yolo_mixed_test/labels/test") / file.name)
                        test_labels_count += 1
                
                # 复制自制测试集（添加前缀避免冲突）
                if os.path.exists("dataset_yolo_private/images/test"):
                    for file in Path("dataset_yolo_private/images/test").glob("*"):
                        new_name = f"private_{file.name}"
                        shutil.copy(file, Path("dataset_yolo_mixed_test/images/test") / new_name)
                        test_images_count += 1
                if os.path.exists("dataset_yolo_private/labels/test"):
                    for file in Path("dataset_yolo_private/labels/test").glob("*"):
                        new_name = f"private_{file.name}"
                        shutil.copy(file, Path("dataset_yolo_mixed_test/labels/test") / new_name)
                        test_labels_count += 1
                
                print(f"混合测试集创建完成：包含{test_images_count}张测试图片，{test_labels_count}个标注文件")
                
                if test_images_count == 0:
                    print("警告：未找到任何测试数据，请确保公开和自制数据集已正确处理")
                    return
                    
            except Exception as e:
                print(f"创建混合测试集时出错：{e}")
                return
            
            if args.stage == 'preprocess':
                print("\n预处理阶段完成！接下来可以执行训练阶段:")
                print("python train.py --compare --stage train")
                return
        
        if args.stage in ['train', 'all']:
            print("\n--- 阶段2: 模型训练 ---")
            # 训练公开数据集模型
            print("训练公开数据集模型...")
            run_train(PUBLIC_CONFIG, "port_public", model=args.model, epochs=args.epochs, batch=args.batch)
            
            # 训练自制数据集模型  
            print("训练自制数据集模型...")
            run_train(PRIVATE_CONFIG, "port_private", model=args.model, epochs=args.epochs, batch=args.batch)
            
            if args.stage == 'train':
                print("\n训练阶段完成！接下来可以执行评估阶段:")
                print("python train.py --compare --stage evaluate")
                return
        
        if args.stage in ['evaluate', 'all']:
            print("\n--- 阶段3: 模型评估 ---")
            # 在两个模型上都运行统一测试集的评估
            print("在统一测试集上评估公开模型...")
            # 这里需要创建统一的测试配置文件
            # 使用已创建的统一测试配置文件
            mixed_test_config = "dataset_yolo_mixed_test/test_config.yaml"
            
            # 评估公开模型在统一测试集上
            print("评估公开模型在统一混合测试集上的性能...")
            # 获取最新的公开模型权重路径
            public_model_paths = list(Path("runs/detect").glob("port_public*/weights/best.pt"))
            if not public_model_paths:
                print("错误：未找到公开模型权重文件，请确保公开数据集训练成功")
                return
            public_model_path = sorted(public_model_paths)[-1]
            
            # 评估自制模型在统一测试集上
            print("评估自制模型在统一混合测试集上的性能...")
            private_model_paths = list(Path("runs/detect").glob("port_private*/weights/best.pt"))
            if not private_model_paths:
                print("错误：未找到自制模型权重文件，请确保自制数据集训练成功")
                return
            private_model_path = sorted(private_model_paths)[-1]
            
            # 运行评估并收集结果
            print(f"使用公开模型: {public_model_path}")
            public_result = run_eval_command(public_model_path, mixed_test_config, "public_on_mixed")
            
            print(f"使用自制模型: {private_model_path}")
            private_result = run_eval_command(private_model_path, mixed_test_config, "private_on_mixed")
            
            # 保存比较结果
            public_results = {
                'model_path': str(public_model_path),
                'status': 'success' if public_result.returncode == 0 else 'failed'
            }
            private_results = {
                'model_path': str(private_model_path),
                'status': 'success' if private_result.returncode == 0 else 'failed'
            }
            
            result_file = save_compare_results(args.compare_dir, public_results, private_results)
            
            print("\n===== 比较完成 =====")
            print(f"详细结果已保存到: {result_file}")
            print("建议对比mAP50和mAP50-95指标来评估两个模型的性能差异")
            
            if args.stage == 'evaluate':
                print("\n评估阶段完成！")
                return

if __name__ == "__main__":
    main()