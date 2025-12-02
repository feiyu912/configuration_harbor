#!/usr/bin/env python3
"""
正确读取U-Net训练结果的脚本
"""

import torch
import numpy as np

def read_epoch_results(epoch_file):
    """读取单个epoch的训练结果（只包含损失信息）"""
    try:
        results = torch.load(epoch_file)
        print(f"=== {epoch_file} 内容 ===")
        print(f"Epoch: {results.get('epoch', 0) + 1}")
        print(f"训练损失: {results.get('train_loss', 0):.4f}")
        print(f"验证损失: {results.get('val_loss', 0):.4f}")
        print(f"包含模型状态字典: {isinstance(results.get('model_state'), dict)}")
        print("注: 此文件只包含损失信息，不包含评估指标(mAP等)")
        print("评估指标仅在训练完成并运行evaluate_model后才会生成\n")
        return results
    except Exception as e:
        print(f"读取{epoch_file}失败: {str(e)}\n")
        return None

def read_training_results():
    """读取完整训练结果（包含评估指标）"""
    try:
        results = torch.load('unet_training_results.pt')
        print("=== unet_training_results.pt 内容 ===")
        print(f"模型: {results.get('model', '未知')}")
        print(f"训练时间(小时): {results.get('training_time_hours', 0):.2f}")
        
        metrics = results.get('metrics', {})
        if metrics:
            print(f"mAP: {metrics.get('mAP', 0):.4f}")
            print(f"精确率: {metrics.get('precision', 0):.4f}")
            print(f"召回率: {metrics.get('recall', 0):.4f}")
            print(f"F1分数: {metrics.get('f1', 0):.4f}")
            print(f"IoU: {metrics.get('iou', 0):.4f}")
            print(f"模型大小(MB): {metrics.get('model_size_mb', 0):.2f}")
            print(f"参数量: {metrics.get('parameters', 0):,}")
        else:
            print("未找到评估指标，可能训练尚未完成评估阶段")
        return results
    except Exception as e:
        print(f"读取unet_training_results.pt失败: {str(e)}")
        print("注: 此文件仅在完整训练和评估完成后才会生成\n")
        return None

def find_all_epoch_files():
    """查找所有保存的epoch文件"""
    import os
    epoch_files = []
    
    for i in range(1, 100):  # 假设最多100个epoch
        file_path = f'unet_epoch{i}.pt'
        if os.path.exists(file_path):
            epoch_files.append(file_path)
    
    return epoch_files

def plot_loss_curve():
    """绘制损失曲线（如果有多个epoch文件）"""
    try:
        import matplotlib.pyplot as plt
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        epoch_files = find_all_epoch_files()
        if len(epoch_files) < 2:
            print("epoch文件数量不足，无法绘制损失曲线")
            return
        
        epochs = []
        train_losses = []
        val_losses = []
        
        for file_path in epoch_files:
            results = torch.load(file_path)
            epochs.append(results.get('epoch', 0) + 1)
            train_losses.append(results.get('train_loss', 0))
            val_losses.append(results.get('val_loss', 0))
        
        # 按epoch排序
        sorted_data = sorted(zip(epochs, train_losses, val_losses))
        epochs, train_losses, val_losses = zip(*sorted_data)
        
        # 绘制曲线
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, 'b-o', label='训练损失')
        plt.plot(epochs, val_losses, 'r-s', label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('U-Net训练损失曲线')
        plt.legend()
        plt.grid(True)
        
        # 保存图像
        plt.savefig('unet_loss_curve.png', dpi=300, bbox_inches='tight')
        print("\n损失曲线已保存到 unet_loss_curve.png")
        plt.close()
        
    except Exception as e:
        print(f"绘制损失曲线失败: {str(e)}")

if __name__ == '__main__':
    print("========== U-Net 训练结果查看 ==========\n")
    
    # 1. 读取最新的epoch文件
    epoch_files = find_all_epoch_files()
    if epoch_files:
        latest_epoch = max(epoch_files, key=lambda x: int(x.split('epoch')[1].split('.')[0]))
        read_epoch_results(latest_epoch)
    else:
        print("未找到保存的epoch文件\n")
    
    # 2. 读取完整训练结果（如果存在）
    read_training_results()
    
    # 3. 绘制损失曲线（如果有足够的数据）
    plot_loss_curve()
    
    print("\n========== 使用说明 ==========")
    print("1. epoch文件只包含损失信息，不包含评估指标")
    print("2. 评估指标(mAP等)仅在模型训练和评估完成后才会生成")
    print("3. 可以运行此脚本查看所有保存的训练信息")