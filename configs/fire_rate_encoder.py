num_classes = 7776

# model settings
loss = [dict(type='PoissonLoss', reduction='sum', loss_weight=1.0)]

model = dict(
    type='FiringRateEncoder',
    backbone=dict(
        type='TimmModels',
        model_name='resnet18',
        features_only=True,
        pretrained=False,
    ),
    head=dict(
        type='BaseHead',
        in_index=-3,
        in_channels=None,
        channels=2048,
        num_classes=num_classes,
        losses=loss
    ),
    # auxiliary_head=None,
    evaluation = dict(metrics=[dict(type='Correlation')])
)

# dataset settings
dataset_type = 'Sensorium'
data_root = '/home/sensorium/sensorium/notebooks/data'
# img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)
size = 256
albu_train_transforms = [
    dict(type='Resize', height=size, width=size, p=1.0),
    dict(type='Normalize'),
    # dict(type='ToTensorV2')
]
train_pipeline = [
    dict(type='LoadImages', image_size=(size, size), channels_first=False, to_RGB=True),
    dict(type='Albumentations', transforms=albu_train_transforms),
    dict(type='ToTensor'),
]
test_pipeline = [
    dict(type='LoadImages', image_size=(size, size), channels_first=False, to_RGB=True),
    dict(type='Albumentations', transforms=albu_train_transforms),
    dict(type='ToTensor'),
]
data = dict(
    train_batch_size=64,  # for single card
    val_batch_size=128,
    test_batch_size=128,
    num_workers=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        feature_dir='static26872-17-20-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip',
        data_keys=['images', 'responses', 'behavior', 'pupil_center'],
        sampler=None,
        pipeline=train_pipeline
        ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        feature_dir='static26872-17-20-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip',
        data_keys=['images', 'responses', 'behavior', 'pupil_center'],
        sampler=None,
        pipeline=test_pipeline,
        ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        feature_dir='static26872-17-20-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip',
        data_keys=['images', 'responses', 'behavior', 'pupil_center'],
        sampler=None,
        pipeline=test_pipeline
    ),
)

# yapf:disable
log = dict(
    project_name='sensorium',
    work_dir='/data2/charon/sensorium',
    exp_name='test',
    logger_interval=50,
    monitor='val_correlation',
    logger=[dict(type='comet', key='0gtDzXpYAKgdFkiEMvw1Y5VcH')],
    checkpoint=dict(type='ModelCheckpoint',
                    filename='{exp_name}-' + \
                             '{val_dice:.3f}',
                    top_k=1,
                    mode='max',
                    verbose=True,
                    save_last=False,
                    ),
    earlystopping=dict(
            mode='max',
            strict=False,
            patience=5,
            min_delta=0.0001,
            check_finite=True,
            verbose=True
    )

)

# yapf:enable
resume_from = None
cudnn_benchmark = True

# optimization
optimization = dict(
    type='epoch',
    max_iters=20,
    optimizer=dict(type='AdamW', lr=4.5e-3, betas=(0.9, 0.999), weight_decay=0.05),
    scheduler=dict(type='CosineAnnealing',
                     # warmup='linear',
                     # warmup_iters=500,
                     # warmup_ratio=1e-6,
                     min_lr=0.0)
)

# runtime settings
# runner = dict(type='IterBasedRunner', max_iters=int(total_iters * 1000))
# checkpoint_config = dict(by_epoch=False, interval=int(total_iters * 1000), save_optimizer=False)
# evaluation = dict(start=0, by_epoch=False, save_best='mDice', interval=min(5000, int(total_iters * 1000)), metric=['imDice', 'mDice'], pre_eval=True)

