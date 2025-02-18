# dataset settings
dataset_type = 'WeedDataset'
data_root = '../../../data/'

# Define class names
classes = ('ABUTH_week_10', 'ABUTH_week_11', 'ABUTH_week_1', 'ABUTH_week_2', 'ABUTH_week_3', 'ABUTH_week_4', 'ABUTH_week_5', 'ABUTH_week_6', 'ABUTH_week_7', 'ABUTH_week_8', 'ABUTH_week_9', 'AMAPA_week_10', 'AMAPA_week_11', 'AMAPA_week_1', 'AMAPA_week_2', 'AMAPA_week_3', 'AMAPA_week_4', 'AMAPA_week_5', 'AMAPA_week_6', 'AMAPA_week_7', 'AMAPA_week_8', 'AMAPA_week_9', 'AMARE_week_10', 'AMARE_week_11', 'AMARE_week_1', 'AMARE_week_2', 'AMARE_week_3', 'AMARE_week_4', 'AMARE_week_5', 'AMARE_week_6', 'AMARE_week_7', 'AMARE_week_8', 'AMARE_week_9', 'AMATA_week_10', 'AMATA_week_11', 'AMATA_week_1', 'AMATA_week_2', 'AMATA_week_3', 'AMATA_week_4', 'AMATA_week_5', 'AMATA_week_6', 'AMATA_week_7', 'AMATA_week_8', 'AMATA_week_9', 'AMBEL_week_10', 'AMBEL_week_11', 'AMBEL_week_1', 'AMBEL_week_2', 'AMBEL_week_3', 'AMBEL_week_4', 'AMBEL_week_5', 'AMBEL_week_6', 'AMBEL_week_7', 'AMBEL_week_8', 'AMBEL_week_9', 'CHEAL_week_10', 'CHEAL_week_11', 'CHEAL_week_1', 'CHEAL_week_2', 'CHEAL_week_3', 'CHEAL_week_4', 'CHEAL_week_5', 'CHEAL_week_6', 'CHEAL_week_7', 'CHEAL_week_8', 'CHEAL_week_9', 'CYPES_week_10', 'CYPES_week_11', 'CYPES_week_1', 'CYPES_week_2', 'CYPES_week_3', 'CYPES_week_4', 'CYPES_week_5', 'CYPES_week_6', 'CYPES_week_7', 'CYPES_week_8', 'CYPES_week_9', 'DIGSA_week_10', 'DIGSA_week_11', 'DIGSA_week_1', 'DIGSA_week_2', 'DIGSA_week_3', 'DIGSA_week_4', 'DIGSA_week_5', 'DIGSA_week_6', 'DIGSA_week_7', 'DIGSA_week_8', 'DIGSA_week_9', 'ECHCG_week_10', 'ECHCG_week_11', 'ECHCG_week_1', 'ECHCG_week_2', 'ECHCG_week_3', 'ECHCG_week_4', 'ECHCG_week_5', 'ECHCG_week_6', 'ECHCG_week_7', 'ECHCG_week_8', 'ECHCG_week_9', 'ERICA_week_10', 'ERICA_week_11', 'ERICA_week_1', 'ERICA_week_2', 'ERICA_week_3', 'ERICA_week_4', 'ERICA_week_5', 'ERICA_week_6', 'ERICA_week_7', 'ERICA_week_8', 'ERICA_week_9', 'PANDI_week_10', 'PANDI_week_11', 'PANDI_week_1', 'PANDI_week_2', 'PANDI_week_3', 'PANDI_week_4', 'PANDI_week_5', 'PANDI_week_6', 'PANDI_week_7', 'PANDI_week_8', 'PANDI_week_9', 'SETFA_week_10', 'SETFA_week_11', 'SETFA_week_1', 'SETFA_week_2', 'SETFA_week_3', 'SETFA_week_4', 'SETFA_week_5', 'SETFA_week_6', 'SETFA_week_7', 'SETFA_week_8', 'SETFA_week_9', 'SETPU_week_10', 'SETPU_week_11', 'SETPU_week_1', 'SETPU_week_2', 'SETPU_week_3', 'SETPU_week_4', 'SETPU_week_5', 'SETPU_week_6', 'SETPU_week_7', 'SETPU_week_8', 'SETPU_week_9', 'SIDSP_week_10', 'SIDSP_week_11', 'SIDSP_week_1', 'SIDSP_week_2', 'SIDSP_week_3', 'SIDSP_week_4', 'SIDSP_week_5', 'SIDSP_week_6', 'SIDSP_week_7', 'SIDSP_week_8', 'SIDSP_week_9', 'SORHA_week_10', 'SORHA_week_11', 'SORHA_week_3', 'SORHA_week_4', 'SORHA_week_5', 'SORHA_week_6', 'SORHA_week_7', 'SORHA_week_8', 'SORHA_week_9', 'SORVU_week_10', 'SORVU_week_11', 'SORVU_week_1', 'SORVU_week_2', 'SORVU_week_3', 'SORVU_week_4', 'SORVU_week_5', 'SORVU_week_6', 'SORVU_week_7', 'SORVU_week_8', 'SORVU_week_9')

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=[0.0, 1],
        rotate_limit=[-45, 45],
        interpolation=1,
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],
        p=0.1),
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),
]

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        },
        skip_img_without_anno=True),
    dict(type='PackDetInputs'),
]

#missing chilo

val_pipeline = [ 
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor')),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor')),
]

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
        ann_file='splitted_labels/train.json',
        data_prefix=dict(img='splitted_images/train'),
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
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
        ann_file='splitted_labels/val.json',
        data_prefix=dict(img='splitted_images/validation'),
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
        ann_file='splitted_labels/test.json',
        data_prefix=dict(img='splitted_images/test'),
        pipeline=test_pipeline,
        test_mode=True,
        backend_args=backend_args))


val_evaluator = [
    dict(
        type='CocoMetric',
        ann_file=data_root + 'splitted_labels/val.json',
        metric=['bbox'],  # Metrics to be evaluated
        classwise=True,
        format_only=False,  # Only format and save the results to coco json file
        backend_args=backend_args
    ),
    dict(
        type='VOCMetric',
        metric=['mAP'],
        iou_thrs=0.5,
        eval_mode='11points'
        )
]


test_evaluator = [
    dict(
        type='CocoMetric',
        ann_file=data_root + 'splitted_labels/test.json',
        metric=['bbox'],  # Metrics to be evaluated
        classwise=True,
        format_only=False,  # Only format and save the results to coco json file
        outfile_prefix='./work_dirs/coco_detection/test',  # The prefix of output json files
    ),
    dict(
        type='VOCMetric',
        metric=['mAP'],
        iou_thrs=0.5,
        eval_mode='11points'
        )
]