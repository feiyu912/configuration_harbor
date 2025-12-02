
# 5070Ti超快训练配置 - 30分钟到1小时完成
_base_ = ['./configs/harbor_port_r50_fpn_1x.py']

# 数据配置 - 极简模式
data = dict(
    samples_per_gpu=16,  # 最大化批次到16
    workers_per_gpu=8,   # 最大化工作线程
    train=dict(
        type='DOTADataset',
        ann_file='data/harbor_port/train/annotations.json',
        img_prefix='data/harbor_port/train/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='RResize', img_scale=(512, 512)),  # 分辨率降到512
            dict(type='RRandomFlip', flip_ratio=0.3, direction='horizontal', version='le90'),  # 仅水平翻转
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],  # 保留归一化，这是必须的
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
        version='le90'),
    val=dict(
        type='DOTADataset',
        ann_file='data/harbor_port/val/annotations.json',
        img_prefix='data/harbor_port/val/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='RResize'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        version='le90'),
    test=dict(
        type='DOTADataset',
        ann_file='data/harbor_port/test/annotations.json',
        img_prefix='data/harbor_port/test/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='RResize'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        version='le90'))

# 模型配置 - 简化模型
model = dict(
    type='RotatedRetinaNet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=3,  # 只使用3个阶段
        out_indices=(0, 1, 2),  # 减少输出层
        frozen_stages=1,
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024],  # 减少输入通道
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=3),  # 减少输出层数
    bbox_head=dict(
        type='RotatedRetinaHead',
        num_classes=3,
        in_channels=256,
        stacked_convs=2,  # 减少卷积层数
        feat_channels=128,  # 减少特征通道
        assign_by_circumhbbox=None,
        anchor_generator=dict(
            type='RotatedAnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=2,  # 减少尺度数量
            ratios=[1.0, 0.5],  # 减少比例
            angles=[0],  # 只检测0度角
            strides=[8, 16, 32]),  # 减少步长数量
        bbox_coder=dict(
            type='DeltaXYWHAOBBoxCoder',
            angle_range='le90',
            norm_factor=None,
            edge_swap=True,
            proj_xy=True,
            target_means=(.0, .0, .0, .0, .0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.11, loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            iou_calculator=dict(type='RBboxOverlaps2D')),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,  # 减少NMS预选择
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='rnms', iou_threshold=0.1),
        max_per_img=500))  # 减少最大检测数

# 优化器配置 - 超快学习率
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)  # 学习率x4
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# 学习率配置 - 超快衰减
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,  # 超短warmup
    warmup_ratio=1.0 / 20,
    step=[4, 8])  # 超快衰减
runner = dict(type='EpochBasedRunner', max_epochs=8)  # 仅8轮

# 检查点和评估
checkpoint_config = dict(interval=2)
evaluation = dict(interval=2, metric='mAP')

# 日志配置
log_config = dict(
    interval=5,  # 超频繁记录
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

# 5070Ti优化 - 启用所有加速
fp16 = dict(loss_scale=256.0)

# 工作目录
work_dir = './work_dirs/harbor_port_r50_fpn_5070ti_ultra_fast'
