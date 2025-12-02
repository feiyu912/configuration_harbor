# RTX 5070Ti Balanced Config - 50 Epochs
model = dict(type='RotatedRetinaNet', backbone=dict(type='ResNet', depth=50))
dataset_type = 'HarborDataset'
data = dict(samples_per_gpu=8, workers_per_gpu=4)
runner = dict(type='EpochBasedRunner', max_epochs=50)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
lr_config = dict(policy='CosineAnnealing', warmup='linear', warmup_iters=500)
fp16 = dict(loss_scale=512.)