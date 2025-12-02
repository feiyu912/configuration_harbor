#!/usr/bin/env python3
"""
Mask R-CNN模型训练脚本
"""

import torch
import time
from deep_learning_models import MaskRCNNSegmentation

def main():
    print("=== Mask R-CNN 模型训练开始 ===")
    
    # 检查GPU
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用！请检查GPU驱动和CUDA安装。本训练仅支持GPU模式。")
    
    print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 创建模型
    print("\n创建Mask R-CNN模型...")
    model = MaskRCNNSegmentation(backbone='resnet50')
    
    # 开始训练
    print("\n开始训练Mask R-CNN模型...")
    start_time = time.time()
    
    try:
        history = model.train_model(
            data_yaml_path='harbor_port_backup/data.yaml',
            epochs=100,  # 完整训练
            patience=20  # 早停耐心值
        )
        
        end_time = time.time()
        training_time = (end_time - start_time) / 3600  # 转换为小时
        
        print(f"\nMask R-CNN训练完成！")
        print(f"训练时间: {training_time:.2f} 小时")
        
        # 评估模型
        print("\n开始评估模型...")
        metrics = model.evaluate_model('harbor_port_backup/data.yaml')
        
        print(f"评估结果:")
        print(f"mAP: {metrics.get('mAP', 0):.4f}")
        print(f"精确率: {metrics.get('precision', 0):.4f}")
        print(f"召回率: {metrics.get('recall', 0):.4f}")
        print(f"F1分数: {metrics.get('f1', 0):.4f}")
        print(f"IoU: {metrics.get('iou', 0):.4f}")
        
        # 保存结果
        results = {
            'model': 'Mask R-CNN (ResNet50-FPN)',
            'training_time_hours': training_time,
            'metrics': metrics,
            'model_size_mb': metrics.get('model_size_mb', 0),
            'parameters': metrics.get('parameters', 0)
        }
        
        torch.save(results, 'maskrcnn_training_results.pt')
        print(f"\n训练结果已保存到: maskrcnn_training_results.pt")
        
    except Exception as e:
        print(f"\n训练过程中出现错误: {str(e)}")
        raise

if __name__ == '__main__':
    main()