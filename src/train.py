import argparse
from pathlib import Path
from ultralytics import YOLO


def train_one(data_yaml: str, model_name: str, imgsz: int, epochs: int, name: str):
    import os
    # 设置离线模式，避免网络检查
    os.environ['ULTRALYTICS_OFFLINE'] = '1'
    
    # 确保每次训练都使用唯一的目录名，避免覆盖
    import time
    unique_name = f"{name}_{int(time.time())}"
    
    model = YOLO(model_name)
    results = model.train(
        data=data_yaml,
        imgsz=imgsz,
        epochs=epochs,
        name=unique_name,
        project="runs/detect",
        exist_ok=False,  # 设置为False，确保创建新目录而不是覆盖
        verbose=True,
    )
    return Path(results.save_dir)


def evaluate(weights_path: Path, data_yaml: str):
    model = YOLO(str(weights_path))
    return model.val(data=data_yaml)


def main():
    parser = argparse.ArgumentParser(description="Train YOLO on port dataset and optionally compare")
    parser.add_argument("--data", required=True, help="path to dataset yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--name", default="port_custom")
    parser.add_argument("--compare-data", default=None, help="optional second dataset yaml")
    args = parser.parse_args()

    run_dir = train_one(args.data, args.model, args.imgsz, args.epochs, args.name)
    best = run_dir / "weights" / "best.pt"
    print(f"Primary training completed: {run_dir}")
    evaluate(best, args.data)

    if args.compare_data:
        # 使用更明确的比较目录名称
        comp_name = f"{args.name}_vs_compare"
        comp_dir = train_one(args.compare_data, args.model, args.imgsz, args.epochs, comp_name)
        comp_best = comp_dir / "weights" / "best.pt"
        print(f"Comparison training completed: {comp_dir}")
        evaluate(comp_best, args.compare_data)


if __name__ == "__main__":
    main()


