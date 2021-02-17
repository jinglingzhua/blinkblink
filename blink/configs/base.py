dataset_type = 'BlinkDataset'
data_root = '/home/zmf/nfs/workspace/blink/data/'
work_dir = '/home/zmf/nfs/workspace/blink/model/test'
batch_size = 256
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_sz = 72
inp_sz = 64
train_pipeline = [
    dict(type='BlinkTransform', crop_sz=crop_sz, inp_sz=inp_sz, training=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img'], meta_keys=('gt_labels',)),
]
test_pipeline = [
    dict(type='BlinkTransform', crop_sz=crop_sz, inp_sz=inp_sz, training=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img'], meta_keys=('gt_labels',)),
]
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        stage='train',
        datadir=data_root+'train',
        repeat=1,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        stage='valid',
        datadir=data_root+'valid',
        pipeline=test_pipeline))
# model settings
model = dict(
    type='BlinkRPN',
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
        type='BlinkHead',
        in_channels=32,
        dense_layers=[32]
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
    warmup_ratio=0.001,
    step=[10, 15])
total_epochs = 20

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
workflow = [('train', 1), ('val', 1)]