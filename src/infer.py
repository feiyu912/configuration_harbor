import argparse
from pathlib import Path
import pandas as pd
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Batch inference and export predictions")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--save-dir", default="outputs")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    images_dir = save_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    results = model.predict(source=args.source, conf=args.conf, save=True, project=str(images_dir), name=".", exist_ok=True)

    records = []
    for r in results:
        img_name = Path(r.path).name
        for b in r.boxes:
            cls_id = int(b.cls.item())
            score = float(b.conf.item())
            x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
            records.append({
                "image": img_name,
                "class_id": cls_id,
                "score": score,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })

    df = pd.DataFrame.from_records(records)
    df.to_csv(save_dir / "predictions.csv", index=False)
    print(f"Saved predictions to {save_dir / 'predictions.csv'} and images to {images_dir}")


if __name__ == "__main__":
    main()


