from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import tempfile
from pathlib import Path


app = FastAPI(title="Port Detection API")
model = None


@app.on_event("startup")
def load_model():
    global model
    weights = Path("runs/detect/port_custom/weights/best.pt")
    if weights.exists():
        model = YOLO(str(weights))
    else:
        # fallback to a coco-pretrained model; user should replace
        model = YOLO("yolov8n.pt")


@app.post("/detect")
async def detect(file: UploadFile = File(...), conf: float = 0.25):
    if model is None:
        return JSONResponse({"error": "model not loaded"}, status_code=500)
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix, delete=True) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp.flush()
        results = model.predict(source=tmp.name, conf=conf)

    r = results[0]
    preds = []
    for b in r.boxes:
        preds.append({
            "class_id": int(b.cls.item()),
            "score": float(b.conf.item()),
            "xyxy": list(map(float, b.xyxy[0].tolist())),
        })
    return {"width": int(r.orig_shape[1]), "height": int(r.orig_shape[0]), "predictions": preds}


