import os
import sys
import traceback

# 设置OpenMP环境变量以避免冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 延迟导入以确保环境变量已设置
print("设置环境变量完成，准备导入YOLO...")

try:
    from ultralytics import YOLO
    print("YOLO库导入成功")
    
    # 创建简单函数
    def train_model():
        print("开始初始化模型...")
        # 加载模型
        model = YOLO('yolov8n-seg.pt')  # 使用更小的模型进行测试
        print("模型加载成功")
        
        print("开始训练...")
        # 使用最简化的训练参数
        results = model.train(
            data='data/data_harbor_backup.yaml',
            epochs=1,
            batch=1,
            workers=0,  # 完全禁用多进程
            device=0,
            pretrained=True,
            verbose=True,
            project='runs/segment',
            name='minimal_test'
        )
        
        print("训练完成！")
        return results
    
    # 在主模块中运行
    if __name__ == '__main__':
        try:
            results = train_model()
            print(f"训练结果: {results}")
        except Exception as e:
            print(f"训练过程中发生错误: {type(e).__name__}: {e}")
            print("详细错误堆栈:")
            traceback.print_exc()
            sys.exit(1)
            
except ImportError as e:
    print(f"导入YOLO库失败: {e}")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"初始化过程中发生错误: {e}")
    traceback.print_exc()
    sys.exit(1)