dataset_type = 'Caltech'
data_root = '/home/zmf/blinkblink/src/facetracker/data/Caltech_WebFaces'
work_dir = '/home/zmf/nfs/workspace/blink/model/test'
batch_size = 256
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
inp_hw = (64,64)
train_pipeline = [
    dict(type='FaceTrackerTransform', inp_hw=inp_hw),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img'], meta_keys=('gt_labels','gt_pts')),
]
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        datadir=data_root,
        bg_fg_ratio=3,
        pipeline=train_pipeline),
)
# model settings
model = dict(
    type='FaceTrackerRPN',
    backbone=dict(
        type='MobileNetV2',
        actModule='relu',
        inverted_residual_setting=[
            [4, 16, 4, 2],
            [4, 32, 4, 2],
            [4, 32, 4, 2],
        ],
        out_indices=[0,1,2]
    ),    
    neck=None,
    rpn_head=dict(
        type='FaceTrackerHead',
        in_channels=32,
        dense_layers=[64]
    ),

)
# model training and testing settings
train_cfg = dict(rpn={})
test_cfg = dict(rpn={})

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.01,
    step=[60, 80])
total_epochs = 100

checkpoint_config = dict(type='BestCheckpointHook', interval=1, max_keep_ckpts=1,
                         watch='loss', policy='min')

# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]