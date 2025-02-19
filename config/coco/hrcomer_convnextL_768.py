# model configuration
norm = 'nn.BatchNorm2d'
act = 'nn.ReLU'
model = dict(
    type='MSPose',
    backbone=dict(
        type='HRCoMer',
        base_net=dict(
            type='ConvNeXtV2',
            depths=[3, 3, 27, 3],
            dims=[192, 384, 768, 1536],
            drop_rate_range=(0.1, 0.3),
            use_ckpt=True,
            freeze_stages=(0, 1, 2, 3),
            pre_weights='pretrained/convnextv2_large_22k_384_ema.pt'
        ),
        hr_adapter=dict(
            type='HRAdapter',
            norm=norm,
            act=act,
            transitions=dict(
                trans1=dict(base_channels=192, hr_channels=64, ratio=1),
                trans2=dict(base_channels=384, hr_channels=96, ratio=1),
                trans3=dict(base_channels=768, hr_channels=192, ratio=1),
                trans4=dict(base_channels=1536, hr_channels=384, ratio=1),
            ),
            hr_modules=dict(
                stage1=dict(
                    num_modules=1,
                    num_branches=1,
                    num_blocks=(1,),
                    num_channels=(64,)),
                stage2=dict(
                    num_modules=1,
                    num_branches=2,
                    num_blocks=(4, 4),
                    num_channels=(48, 96)),
                stage3=dict(
                    num_modules=4,
                    num_branches=3,
                    num_blocks=(4, 4, 4, 4),
                    num_channels=(48, 96, 192)),
                stage4=dict(
                    num_modules=3,
                    num_branches=4,
                    num_blocks=(4, 4, 4, 4),
                    num_channels=(48, 96, 192, 384))
            ),
            pre_weights='pretrained/hrnet_w48-8ef0771d.pth'
        )
    ),
    in_channels=[48, 96, 192, 384],
    num_joints=17,
    norm=norm,
    act=act
)

# trainer configuration
trainer = dict(
    model=model,
    matcher=dict(
        cost=dict(
            cost_cls=dict(type='BinaryFocalCost', gamma=2, alpha=0.25, weight=1.),
            cost_joint=dict(type='JointL1Cost', weight=4.),
            cost_limb=dict(type='LimbL1Cost', weight=2.)
        ),
        output_size=192,
        num_scales=3
    ),
    criterion=dict(
        loss_cls=dict(type='BinaryFocalLoss', gamma=2, alpha=0.25, weight=1 / 4 * 2.25),
        loss_joint=dict(type='JointL1Loss', weight=2e-3),
        loss_limb=dict(type='LimbL1Loss', weight=1e-3),
        loss_oks=dict(type='OksLoss', weight=1e-3),
        loss_hms=dict(type='HeatmapFocalLoss', weight=1e-3)
    )
)

# evaluation configuration
test_scale_factors = [1]
eval_cfg = dict(
    ratio=4.0,
    val_th=0.1,
    max_num_people=30,
    test_scale_factors=test_scale_factors
)

# data-set configuration
data_root = 'YOUR_DATA_DIR'
data_cfg = dict(
    input_size=768,
    output_size=192,
    use_nms=False,
    soft_nms=False,
    oks_thr=0.9
)
train_pipeline = [
    dict(type='LoadImageFromFile', channel_order='bgr'),
    dict(type='RandomFlip', flip_prob=0.5),
    dict(type='RandomAffine',
         rot_factor=30,
         scale_factor=[0.75, 1.5],
         scale_type='short',
         trans_factor=40),
    dict(type='FormatGroundTruth', max_num_people=30),
    dict(type='NormalizeImage',
         mean=[123.675, 116.28, 103.53],
         std=[58.395, 57.12, 57.375],
         to_rgb=True),
    dict(type='Collect',
         keys=['image', 'masks', 'target_heatmaps', 'target_joints', 'target_areas', 'target_sizes'],
         meta_keys=[])
]
val_pipeline = [
    dict(type='LoadImageFromFile', channel_order='bgr'),
    dict(type='ResizeAlign', size_divisor=64, test_scale_factors=test_scale_factors),
    dict(type='NormalizeImage',
         mean=[123.675, 116.28, 103.53],
         std=[58.395, 57.12, 57.375],
         to_rgb=True),
    dict(type='Collect',
         keys=['image'],
         meta_keys=['image_file', 'base_size', 'center', 'scale'])
]
set_cfg = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='COCOPose',
        ann_file='{}/annotations/person_keypoints_train2017.json'.format(data_root),
        img_prefix='{}/train2017/'.format(data_root),
        pipeline=train_pipeline),
    val=dict(
        type='COCOPose',
        ann_file='{}/annotations/person_keypoints_val2017.json'.format(data_root),
        img_prefix='{}/val2017/'.format(data_root),
        pipeline=val_pipeline),
    test=dict(
        type='COCOPose',
        ann_file='{}/annotations/image_info_test-dev2017.json'.format(data_root),
        img_prefix='{}/test2017/'.format(data_root),
        pipeline=val_pipeline)
)

# solver
solver = dict(
    optimizer=dict(
        type='AdamW',
        lr=5e-4,
        weight_decay=1e-4,
        lr_decay={
            'backbone.base_net': 0.1,
        }
    ),
    lr_scheduler=dict(
        warmup_iters=500,
        warmup_ratio=1e-3,
        milestones=[40, 60],
        gamma=0.1
    ),
    with_autocast=True,
    total_epochs=70,
    eval_interval=5,   # epoch
    log_interval=50,   # iter
    log_loss=['loss_cls', 'loss_joint', 'loss_limb', 'loss_oks', 'loss_hms']
)
