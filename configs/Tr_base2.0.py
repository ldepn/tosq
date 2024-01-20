classes = ('background', 'shelf', 'door', 'wall', 'box', 'freezer',
           'window', 'cup', 'bottle', 'jar', 'bowl', 'eyeglass')

palette = [[0, 0, 0], [120, 120, 70], [120, 120, 120], [224, 5, 255], [4, 250, 7], [6, 230, 230],
           [204, 255, 4], [255, 51, 7], [150, 5, 61], [235, 255, 7], [204, 5, 255], [140, 140, 140]]

dataset_type = 'BaseSegDataset'
data_root = '/root/autodl-tmp/Trans10K_cls12'

crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # dict(type='RandomResize', scale=(4096, crop_size[0]), ratio_range=(0.75, 1.5), keep_ratio=True),
    dict(type='RandomResize', scale=crop_size, ratio_range=(0.75, 1.5), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=(4096, crop_size[0]), keep_ratio=True),
    # dict(type='ResizeToMultiple', size_divisor=32),
    dict(type='Resize', scale=crop_size, keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')]

EpochBased = False
train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True) if EpochBased else dict(type='InfiniteSampler', shuffle=True),
    drop_last=False,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_suffix='.jpg',
        seg_map_suffix='_mask.png',
        metainfo=dict(classes=classes, palette=palette),
        data_prefix=dict(img_path='train/images', seg_map_path='train/masks_12'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    drop_last=False,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_suffix='.jpg',
        seg_map_suffix='_mask.png',
        metainfo=dict(classes=classes, palette=palette),
        data_prefix=dict(img_path='validation/images', seg_map_path='validation/masks_12'),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    drop_last=False,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_suffix='.jpg',
        seg_map_suffix='_mask.png',
        metainfo=dict(classes=classes, palette=palette),
        data_prefix=dict(img_path='test/images', seg_map_path='test/masks_12'),
        test_mode=True,
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'])
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'],
                      format_only=False, output_dir=None, prefix=None)
tta_model = dict(type='SegTTAModel')

default_scope = 'mmseg'
env_cfg = dict(cudnn_benchmark=True, mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
               dist_cfg=dict(backend='nccl'))

visualizer = dict(type='SegLocalVisualizer',
                  vis_backends=[dict(type='LocalVisBackend')],
                  name='visualizer',
                  save_dir='',
                  classes=classes, palette=palette, alpha=0.8)

train_cfg = dict(by_epoch=True, max_epochs=50) if EpochBased \
    else dict(by_epoch=False, max_iters=20000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=200, interval_exp_name=200000, log_metric_by_epoch=EpochBased),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=EpochBased, interval=1, save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=False, interval=200))
custom_hooks = [    
    # dict(type='EMAHook', begin_iter=500, priority='NORMAL')
]
log_processor = dict(window_size=10, by_epoch=EpochBased, custom_cfg=None, num_digits=3, log_with_hierarchy=True)
log_level = 'INFO'
load_from = ''
resume = False
experiment_name = ''
work_dir = ''
randomness = dict(seed=2023, deterministic=False)

# optimizer
optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type='AdamW', lr=0.0001, weight_decay=5e-3),
                     accumulative_counts=1,
                     paramwise_cfg=dict(bias_decay_mult=0., norm_decay_mult=0.,
                                        custom_keys={'cls_embed': dict(decay_mult=0.),
                                                     'head': dict(lr_mult=10.)
                                                     }),
                     clip_grad=dict(type='norm', max_norm=15, norm_type=2, error_if_nonfinite=True))
# learning policy
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, by_epoch=False, begin=0, end=500),  # warm-up
    dict(type='PolyLR', eta_min=0, power=0.9, begin=0, end=20000, by_epoch=False)
]

norm_cfg = dict(type='LN', requires_grad=True, eps=1e-6)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        size=crop_size,
        pad_val=0,
        seg_pad_val=255,
        # test_cfg=dict(size_divisor=32)
    ),
    init_cfg=None,
    backbone=dict(
        type='MixVisionTransformer',
        init_cfg=dict(type='Pretrained', prefix='backbone.',
                      checkpoint='https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b2_512x512_160k_ade20k/segformer_mit-b2_512x512_160k_ade20k_20210726_112103-cbd414ac.pth'),
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 4, 6, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.1,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    decode_head=dict(
        type='SegTrueHead',
        init_cfg=None,
        query_embed_dims=128,
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        num_stages=4,
        num_heads=[8, 8, 8, 8],
        qkv_bias=True,
        channels=64,
        dropout_ratio=0.1,  # pred_head dropout ratio
        drop_rate=0.1,  # attn output and ffn dropout ratio
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        mlp_ratio=4,
        act_cfg=dict(type='GELU'),
        norm_cfg=norm_cfg,
        num_classes=12,
        align_corners=False,
        # sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000),
        loss_decode=[dict(type='DiceLoss', loss_weight=1.1), dict(type='CrossEntropyLoss', use_sigmoid=False,
                                                                  loss_weight=1.0, avg_non_ignore=True)]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
