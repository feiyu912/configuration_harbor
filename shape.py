import torch
from ultralytics import YOLO

# 1. 加载预训练模型
# 注意：YOLO会自动下载 yolov8m-seg.pt 如果本地不存在
try:
    model = YOLO('yolov8m-seg.pt')
except Exception as e:
    print(f"Error loading model: {e}")
    # 如果加载失败，可以尝试直接加载权重文件
    # model = torch.load('yolov8m-seg.pt')

# 2. 遍历模型的命名参数
print(f"--- 权重文件: yolov8m-seg.pt 的参数形状 ---")
print(f"{'Parameter Name':<60} | {'Shape'}")
print("-" * 80)

# 使用 model.model.named_parameters() 获取所有可训练参数
# .model 属性是底层的 nn.Module
for name, param in model.model.named_parameters():
    # 形状是一个 torch.Size 对象，我们将其转换为字符串
    shape_str = str(list(param.shape))
    print(f"{name:<60} | {shape_str}")

# 也可以查看模型的结构 (Summary)
# print("\n--- 模型结构摘要 ---")
# model.model.info(verbose=False)