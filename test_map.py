import torch

# 读取评估结果
results = torch.load('unet_epoch1.pt')
metrics = results.get('metrics', {})
print(f"mAP: {metrics.get('mAP', 0):.4f}")
print(f"精确率: {metrics.get('precision', 0):.4f}")
print(f"召回率: {metrics.get('recall', 0):.4f}")
print(f"F1分数: {metrics.get('f1', 0):.4f}")
print(f"IoU: {metrics.get('iou', 0):.4f}")