# WeedSwin Configuration File
# This is a complete configuration for training and testing the WeedSwin model
# WeedSwin uses Swin Transformer backbone with DINO detection head

# Dataset settings
dataset_type = 'CocoDataset'  # Using COCO format for weed detection
data_root = 'data/'  # Update this path to your dataset location

# Define weed classes (174 classes: 16 species × 11 growth stages)
classes = ('ABUTH_week_10', 'ABUTH_week_11', 'ABUTH_week_1', 'ABUTH_week_2', 'ABUTH_week_3', 'ABUTH_week_4', 'ABUTH_week_5', 'ABUTH_week_6', 'ABUTH_week_7', 'ABUTH_week_8', 'ABUTH_week_9', 'AMAPA_week_10', 'AMAPA_week_11', 'AMAPA_week_1', 'AMAPA_week_2', 'AMAPA_week_3', 'AMAPA_week_4', 'AMAPA_week_5', 'AMAPA_week_6', 'AMAPA_week_7', 'AMAPA_week_8', 'AMAPA_week_9', 'AMARE_week_10', 'AMARE_week_11', 'AMARE_week_1', 'AMARE_week_2', 'AMARE_week_3', 'AMARE_week_4', 'AMARE_week_5', 'AMARE_week_6', 'AMARE_week_7', 'AMARE_week_8', 'AMARE_week_9', 'AMATA_week_10', 'AMATA_week_11', 'AMATA_week_1', 'AMATA_week_2', 'AMATA_week_3', 'AMATA_week_4', 'AMATA_week_5', 'AMATA_week_6', 'AMATA_week_7', 'AMATA_week_8', 'AMATA_week_9', 'AMBEL_week_10', 'AMBEL_week_11', 'AMBEL_week_1', 'AMBEL_week_2', 'AMBEL_week_3', 'AMBEL_week_4', 'AMBEL_week_5', 'AMBEL_week_6', 'AMBEL_week_7', 'AMBEL_week_8', 'AMBEL_week_9', 'CHEAL_week_10', 'CHEAL_week_11', 'CHEAL_week_1', 'CHEAL_week_2', 'CHEAL_week_3', 'CHEAL_week_4', 'CHEAL_week_5', 'CHEAL_week_6', 'CHEAL_week_7', 'CHEAL_week_8', 'CHEAL_week_9', 'CYPES_week_10', 'CYPES_week_11', 'CYPES_week_1', 'CYPES_week_2', 'CYPES_week_3', 'CYPES_week_4', 'CYPES_week_5', 'CYPES_week_6', 'CYPES_week_7', 'CYPES_week_8', 'CYPES_week_9', 'DIGSA_week_10', 'DIGSA_week_11', 'DIGSA_week_1', 'DIGSA_week_2', 'DIGSA_week_3', 'DIGSA_week_4', 'DIGSA_week_5', 'DIGSA_week_6', 'DIGSA_week_7', 'DIGSA_week_8', 'DIGSA_week_9', 'ECHCG_week_10', 'ECHCG_week_11', 'ECHCG_week_1', 'ECHCG_week_2', 'ECHCG_week_3', 'ECHCG_week_4', 'ECHCG_week_5', 'ECHCG_week_6', 'ECHCG_week_7', 'ECHCG_week_8', 'ECHCG_week_9', 'ERICA_week_10', 'ERICA_week_11', 'ERICA_week_1', 'ERICA_week_2', 'ERICA_week_3', 'ERICA_week_4', 'ERICA_week_5', 'ERICA_week_6', 'ERICA_week_7', 'ERICA_week_8', 'ERICA_week_9', 'PANDI_week_10', 'PANDI_week_11', 'PANDI_week_1', 'PANDI_week_2', 'PANDI_week_3', 'PANDI_week_4', 'PANDI_week_5', 'PANDI_week_6', 'PANDI_week_7', 'PANDI_week_8', 'PANDI_week_9', 'SETFA_week_10', 'SETFA_week_11', 'SETFA_week_1', 'SETFA_week_2', 'SETFA_week_3', 'SETFA_week_4', 'SETFA_week_5', 'SETFA_week_6', 'SETFA_week_7', 'SETFA_week_8', 'SETFA_week_9', 'SETPU_week_10', 'SETPU_week_11', 'SETPU_week_1', 'SETPU_week_2', 'SETPU_week_3', 'SETPU_week_4', 'SETPU_week_5', 'SETPU_week_6', 'SETPU_week_7', 'SETPU_week_8', 'SETPU_week_9', 'SIDSP_week_10', 'SIDSP_week_11', 'SIDSP_week_1', 'SIDSP_week_2', 'SIDSP_week_3', 'SIDSP_week_4', 'SIDSP_week_5', 'SIDSP_week_6', 'SIDSP_week_7', 'SIDSP_week_8', 'SIDSP_week_9', 'SORHA_week_10', 'SORHA_week_11', 'SORHA_week_3', 'SORHA_week_4', 'SORHA_week_5', 'SORHA_week_6', 'SORHA_week_7', 'SORHA_week_8', 'SORHA_week_9', 'SORVU_week_10', 'SORVU_week_11', 'SORVU_week_1', 'SORVU_week_2', 'SORVU_week_3', 'SORVU_week_4', 'SORVU_week_5', 'SORVU_week_6', 'SORVU_week_7', 'SORVU_week_8', 'SORVU_week_9')

backend_args = None

# Model configuration
num_levels = 4

model = dict(
    type='DINO',
    num_queries=300,
    num_feature_levels=num_levels,
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    # WeedSwin uses Swin Transformer as backbone instead of ResNet
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 4],  # Deep architecture for better feature extraction
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=True,
        convert_weights=True),
    neck=dict(
        type='ChannelMapper',
        in_channels=[192, 384, 768, 1536],  # Match Swin Transformer output channels
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=num_levels),
    encoder=dict(
        num_layers=4,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=num_levels, dropout=0.0),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0))),
    decoder=dict(
        num_layers=4,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_cfg=dict(embed_dims=256, num_levels=num_levels, dropout=0.0),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=0.0,
        temperature=20),
    bbox_head=dict(
        type='DINOHead',
        num_classes=174,  # 174 weed classes
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)])),
    test_cfg=dict(max_per_img=100))

# Data pipeline
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [dict(
                type='RandomChoiceResize',
                scales=[(320, 480), (352, 512), (384, 544), (416, 576),
                        (448, 608), (480, 640)],
                keep_ratio=True)],
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(280, 1400), (320, 1400), (360, 1400)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(256, 384),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(320, 480), (352, 512), (384, 544), (416, 576),
                            (448, 608), (480, 640)],
                    keep_ratio=True)]]),
    dict(type='PackDetInputs')]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))]

# Dataloader configuration
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='annotations/train.json',  # Update with your annotation file
        data_prefix=dict(img='train/'),  # Update with your image directory
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=False),
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='annotations/val.json',
        data_prefix=dict(img='val/'),
        pipeline=test_pipeline,
        test_mode=True,
        backend_args=backend_args))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='annotations/test.json',
        data_prefix=dict(img='test/'),
        pipeline=test_pipeline,
        test_mode=True,
        backend_args=backend_args))

# Evaluation metrics
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val.json',
    metric=['bbox'],
    classwise=True,
    format_only=False,
    backend_args=backend_args)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/test.json',
    metric=['bbox'],
    classwise=True,
    format_only=False,
    backend_args=backend_args)

# Training configuration
max_epochs = 12

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimizer configuration (adjusted for deeper Swin Transformer)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.05),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)}))

# Learning rate scheduler
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[11],
        gamma=0.1)]

auto_scale_lr = dict(base_batch_size=16)

# Runtime configuration
default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
work_dir = './work_dirs/weedswin'

# Pretrained weights
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/dino/dino-5scale_swin-l_8xb2-12e_coco/dino-5scale_swin-l_8xb2-12e_coco_20230228_072924-a654145f.pth'
resume = False
