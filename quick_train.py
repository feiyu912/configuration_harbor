import os
import sys

try:
    from ultralytics import YOLO
    print("YOLO库导入成功")
    
    # 使用已经下载的模型
    model = YOLO('yolov8m-seg.pt')
    print("模型加载成功")
    
    print("开始快速训练测试...")
    # 只运行1个epoch进行测试
    results = model.train(
        data='data/data_harbor_backup.yaml',
        epochs=1,
        batch=8,
        imgsz=640,
        device=0,
        workers=0,
        pretrained=True,
        verbose=True,
        project='runs/segment',
        name='quick_test_training'
    )
    
    print("\n快速训练测试完成！")
    print(f"训练结果目录: {model.trainer.save_dir}")
    
    # 检查是否有结果生成
    if os.path.exists(model.trainer.save_dir):
        print("\n训练结果目录内容:")
        for item in os.listdir(model.trainer.save_dir):
            print(f"  - {item}")
    
    # 打印一些训练指标
    if hasattr(results, 'metrics'):
        print("\n训练指标:")
        for key, value in results.metrics.items():
            print(f"  {key}: {value}")
    
    print("\n快速训练测试成功完成！")
    
except Exception as e:
    print(f"训练过程中发生错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)