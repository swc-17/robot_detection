_base_ = [
    '../_base_/datasets/voc0712.py',
    './yolox_nano_8x8_300e_coco.py'
]


optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,  # same as burn-in in darknet
    warmup_ratio=0.1,
    step=[17, 22])
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=30)
evaluation = dict(interval=1, metric=['mAP'])

custom_imports = dict(
    imports=['datasets'],
    allow_failed_imports=False)
