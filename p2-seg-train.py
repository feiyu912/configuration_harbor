import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO

def main():
    # 1. 结构 + 权重 一键加载（只加载对得上的层）
    model = YOLO('ultralytics/ultralytics/cfg/models/v8/yolov8m-seg-p2.yaml').load('yolov8m-seg.pt')

    # 2. 直接开训，epoch 给够，patience 加大
    model.train(
        data='harbor_port_backup/data.yaml',
        epochs=300,          # 之前 150 不够
        patience=50,         # 早停别误杀
        batch=8,
        imgsz=768,
        device=0,
        workers=4,
        optimizer='SGD',
        lr0=0.01,
        cos_lr=True,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.3,
        close_mosaic=15,
        project='runs/segment_p2',
        name='harbor_merged_p2'
    )

if __name__ == '__main__':
    main()