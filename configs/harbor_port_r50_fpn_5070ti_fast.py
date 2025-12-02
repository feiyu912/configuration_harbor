
# 5070Ti快速训练配置 - 30分钟到1小时完成
_base_ = ['./configs/harbor_port_r50_fpn_1x.py']

# 数据配置 - 减少数据增强以加速
data = dict(
    samples_per_gpu=8,  # 增大批次到8
    workers_per_gpu=6,  # 增加工作线程
    train=dict(
        type='DOTADataset',
        ann_file='data/harbor_port/train/annotations.json',
        img_prefix='data/harbor_port/train/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='RResize', img_scale=(768, 768)),  # 降低分辨率到768
            dict(type='RRandomFlip', flip_ratio=0.5, direction='horizontal', version='le90'),  # 只保留水平翻转
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
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
                img_scale=(768, 768),  # 验证也使用768
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
                img_scale=(768, 768),
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

# 优化器配置 - 使用更大学习率加速收敛
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)  # 学习率翻倍
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# 学习率配置 - 快速衰减
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,  # 减少warmup轮数
    warmup_ratio=1.0 / 10,  # 加快warmup
    step=[6, 10])  # 提前衰减
runner = dict(type='EpochBasedRunner', max_epochs=12)  # 12轮足够

# 检查点和评估 - 更频繁保存
checkpoint_config = dict(interval=2)  # 每2轮保存
evaluation = dict(interval=2, metric='mAP')  # 每2轮评估

# 日志配置 - 更频繁记录
log_config = dict(
    interval=10,  # 每10个批次记录
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

# 5070Ti特定优化 - 启用FP16加速
fp16 = dict(loss_scale=512.0)

# 工作目录
work_dir = './work_dirs/harbor_port_r50_fpn_5070ti_fast'
