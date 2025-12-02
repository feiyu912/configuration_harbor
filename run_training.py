import logging
import sys
import os

# 配置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

print("当前工作目录:", os.getcwd())

# 尝试导入并运行训练
try:
    print("导入YOLO库...")
    from ultralytics import YOLO
    print("YOLO库导入成功")
    
    print("加载模型...")
    model = YOLO('yolov8m-seg.pt')
    print("模型加载成功")
    
    print("检查数据集配置...")
    # 检查配置文件是否存在
    config_path = 'data/data_private.yaml'
    if os.path.exists(config_path):
        print(f"配置文件存在: {config_path}")
        with open(config_path, 'r') as f:
            print("配置文件内容:")
            print(f.read())
    else:
        print(f"配置文件不存在: {config_path}")
    
    print("开始训练...")
    results = model.train(
        data=config_path,
        epochs=100,
        batch=16,
        imgsz=640,
        device=0,
        workers=4,
        pretrained=True,
        optimizer='Adam',
        patience=10,
        freeze=10,
        verbose=True,
        project='runs/segment',
        name='private_dataset_training'
    )
    print("训练完成")
    print(f"训练结果: {results}")
    
except Exception as e:
    print(f"训练过程中发生错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)