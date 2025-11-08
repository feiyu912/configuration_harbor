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
    parser = argparse.ArgumentParser(description="Train YOLO on port dataset")
    parser.add_argument("--data", required=True, help="path to dataset yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--name", default="port_custom")
    parser.add_argument("--dataset-type", choices=["public", "private"], default="public", help="数据集类型")
    args = parser.parse_args()

    if args.epochs == 0:
        # 0个epoch表示只评估不训练
        print(f"直接评估模型: {args.model}")
        model = YOLO(args.model)
        results = model.val(data=args.data)
        print(f"评估结果: {results}")
    else:
        run_dir = train_one(args.data, args.model, args.imgsz, args.epochs, args.name)
        best = run_dir / "weights" / "best.pt"
        print(f"训练完成: {run_dir}")
        evaluate(best, args.data)


if __name__ == "__main__":
    main()


