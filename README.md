## Port Operations Object Detection (Ships, Containers, Cranes)

This project provides an end-to-end pipeline for port-scene object detection using YOLOv8: data preparation and augmentation, training/evaluation on public vs custom datasets, inference (CLI/API), and a Streamlit dashboard for visualization.

### Classes
- 0: ship
- 1: container
- 2: crane

### Project Structure
```
configs/
  port.yaml               # YOLO dataset config (custom)
src/
  data_prep.py           # Split dataset, create YOLO layout, augment
  train.py               # Train and evaluate models
  infer.py               # CLI batch inference and CSV export
  api.py                 # FastAPI inference service
app/
  streamlit_app.py       # Dashboard to browse predictions
requirements.txt         # Python dependencies
```

### Installation
```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

PyTorch install may vary by CUDA version. If needed, install from the official index before installing `ultralytics`.

### Data Preparation
Option 1 (recommended): Use a specialized dataset like SeaShips (VOC format)
1) 一键准备（下载/解压/转换/切分，可传网络 URL 或本地 ZIP/目录）：
```bash
python src/get_dataset.py --source <SeaShips_ZIP_URL_or_local_path> --augment --aug-per-image 2
```

或手动步骤：
下载并解压 SeaShips（或其它 VOC 标注数据集），假设目录如下：
```
seaships/
  JPEGImages/          # 图片
  Annotations/         # VOC XML
```
2) 转为 YOLO 原始集：
```bash
python src/convert_voc_to_yolo.py --voc-images seaships/JPEGImages \
  --voc-annots seaships/Annotations --out dataset_raw
```

Option 2: Use your own labeled data. Organize or export your annotations in YOLO format or COCO/VOC and convert externally. Place raw images and labels here:
```
dataset_raw/
  images/
  labels/                # YOLO .txt per image (class x y w h, normalized)
```

Create a working dataset with train/val/test splits and augmentations:
```bash
python src/data_prep.py \
  --raw-dir dataset_raw \
  --out-dir dataset_yolo \
  --val-ratio 0.15 --test-ratio 0.15 \
  --augment --aug-per-image 2
```

Update `configs/port.yaml` paths if you place the dataset elsewhere.

### Training and Evaluation
Train on custom dataset:
```bash
python src/train.py --data configs/port.yaml --epochs 50 --imgsz 960 \
  --model yolov8n.pt --name custom_port
```

Optionally compare against a public dataset (provide another YAML):
```bash
python src/train.py --data configs/port.yaml --compare-data path/to/public.yaml \
  --epochs 50 --imgsz 960 --model yolov8n.pt --name compare_run
```

Metrics (mAP50-95, PR curves) and confusion matrices are saved under `runs/detect/*/`.

### Inference (CLI)
```bash
python src/infer.py --weights runs/detect/custom_port/weights/best.pt \
  --source sample_images --save-dir outputs --conf 0.25
```

This writes annotated images and a `predictions.csv` with boxes, classes, and scores.

### FastAPI Service
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

POST an image to `/detect` to receive predictions.

### Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py -- \
  --images-dir outputs/images --preds-csv outputs/predictions.csv
```

### Notes on Reasonable Data and Rigor
- Ensure data sources are compliant and relevant to port operations.
- The pipeline enforces splits, augmentations, and reports comprehensive metrics.
- For independent analysis, compare custom vs public datasets via `--compare-data` and discuss differences in model behavior.


