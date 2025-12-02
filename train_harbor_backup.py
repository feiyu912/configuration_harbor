import os
import sys

# 设置环境变量以解决OpenMP冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def main():
    try:
        from ultralytics import YOLO
        print("YOLO库导入成功")

        # 使用已经下载的模型
        model = YOLO('yolo11x-seg.pt')

        print("模型加载成功")

        print("开始训练...")
        # 使用harbor_port_backup数据集进行训练，降低batch size和workers以解决Windows多进程问题
        results = model.train(
            data='data/data_harbor_backup.yaml',
            epochs=100,
            batch=8,  # 5070 Ti 16 GB 安全值
            imgsz=768,  # 分辨率放大
            device=0,
            workers=4,
            pretrained=True,
            optimizer='SGD',
            lr0=0.01,
            cos_lr=True,
            mosaic=1.0,
            mixup=0.3,
            copy_paste=0.5,
            auto_augment='randaugment',
            verbose=True,
            project='runs/segment_p2',
            name='harbor_opt'
        )

        print("\n训练完成！")
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

        print("\n训练过程已完成。")

    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    # 对于Windows系统，确保在主模块中执行训练
    main()