# RTX 5070Ti 50轮平衡训练配置
# 针对港口旋转目标检测优化

model = dict(
    type='RotatedRetinaNet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RotatedRetinaHead',
        num_classes=3,  # 船只、集装箱、起重机
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='RotatedAnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[1.0, 0.5, 2.0],
            angles=[0, 45, 90, 135],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHAOBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0))))

# 数据集配置
dataset_type = 'HarborDataset'
data_root = 'data/harbor_port/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(768, 768)),  # RTX 5070Ti优化分辨率
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='RRandomRotate', rotate_ratio=0.5, angles=[-45, 0, 45]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RResize', img_scale=(768, 768)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img'])]

# 数据配置
data = dict(
    samples_per_gpu=8,      # RTX 5070Ti优化批次大小
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + 'val/',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/test.json',
        img_prefix=data_root + 'test/',
        pipeline=val_pipeline))

# 训练配置 - 50轮优化
runner = dict(type='EpochBasedRunner', max_epochs=50)  # 50轮训练
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# 学习率调度 - 余弦退火
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    min_lr_ratio=1e-5)

# 混合精度训练
fp16 = dict(loss_scale=512.)

# 检查点和评估
checkpoint_config = dict(interval=10)  # 每10轮保存一次
evaluation = dict(interval=5, metric='mAP')  # 每5轮评估一次

# 日志配置
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')])

# 系统配置
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# 自定义设置
work_dir = 'work_dirs/harbor_port_r50_fpn_5070ti_50epochs'
