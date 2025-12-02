from ultralytics import YOLO

# 加载两个模型
model_old = YOLO('yolov8m-seg.pt')
model_new = YOLO('ultralytics/ultralytics/cfg/models/v8/yolov8m-seg-p2.yaml')

print(f"{'OLD Model Layer':<50} | {'NEW Model Layer':<50} | {'Status'}")
print("-" * 120)

# 获取参数列表
# 注意：这里我们只对比 model.named_parameters()
old_params = dict(model_old.model.named_parameters())
new_params = dict(model_new.model.named_parameters())

for name_new, param_new in new_params.items():
    status = "MISMATCH / NEW"
    name_old_display = "---"

    if name_new in old_params:
        param_old = old_params[name_new]
        if param_old.shape == param_new.shape:
            status = "MATCH (Loaded)"
            name_old_display = name_new
        else:
            status = "SHAPE MISMATCH (Re-init)"
            name_old_display = f"{name_new} (Shape diff)"

    # 只打印前几层和典型的 Head 层，避免刷屏
    if "backbone" in name_new or "0.cv" in name_new or "head" in name_new:
        print(f"{name_old_display:<50} | {name_new:<50} | {status}")