# model settings
# 10 classes for cifar10
num_classes = 10
# loss function, set multiple losses if needed, allow weighted sum
loss = [dict(type='TorchLoss', loss_name='CrossEntropyLoss', loss_weight=1.0)]

model = dict(
    # Base Encoder Decoder Architecture (Backbone+Head)
    type='BaseEncoderDecoder',
    # Visual Backbone from Timm Models library
    backbone=dict(
        type='TimmModels',
        model_name='resnet50',
        features_only=False,
        remove_fc=True,
        pretrained=False
    ),
    # Linear Head for classification
    head=dict(
        type='BaseHead',
        in_index=-1,
        dropout=0.0,
        num_classes=num_classes,
        channels=None,
        losses=loss
    ),
    # auxiliary_head, only work in training for auxiliary loss
    # auxiliary_head=None,
    # metrics for evaluation, allow multiple metrics
    evaluation = dict(metrics=[dict(type='TorchMetrics', metric_name='Accuracy',
                                    prob=True)])
)

# dataset settings
dataset_type = 'TorchVision'
dataset_name = 'CIFAR10'

# training data preprocess pipeline
train_pipeline = [dict(type='LoadImages'),
                  dict(type='Albumentations', transforms=[
                                                          dict(type='PadIfNeeded', min_height=40, min_width=40, p=1.0),
                                                          dict(type='RandomCrop', height=32, width=32, p=1.0),
                                                          dict(type='HorizontalFlip', p=0.5),
                                                          dict(type='Normalize', mean=[0.4914, 0.4822, 0.4465],
                                                               std=[0.2023, 0.1994, 0.2010], p=1.0)]),
                  dict(type='ToTensor')]

# validation data preprocess pipeline
test_pipeline = [dict(type='LoadImages'),
                 dict(type='Albumentations', transforms=[dict(type='Normalize', mean=[0.4914, 0.4822, 0.4465],
                                                              std=[0.2023, 0.1994, 0.2010], p=1.0)]),
                 dict(type='ToTensor')]

data = dict(
    train_batch_size=128,  # for single card
    val_batch_size=256,
    test_batch_size=256,
    num_workers=4,
    train=dict(
        type=dataset_type,
        dataset_name=dataset_name,
        train=True,
        sampler=None,  # None is default sampler, set to RandomSampler/DistributedSampler
        pipeline=train_pipeline
        ),
    val=dict(
        type=dataset_type,
        dataset_name=dataset_name,
        train=False,
        sampler='SequentialSampler',
        pipeline=test_pipeline,
        ),
    test=dict(
        type=dataset_type,
        dataset_name=dataset_name,
        train=False,
        sampler='SequentialSampler',
        pipeline=test_pipeline
    ),
)

# yapf:disable
log = dict(
    # project name, used for cometml
    project_name='cifar_test',
    # work directory, used for saving checkpoints and loggings
    work_dir='/data2/charon/test/ckpts',
    # explicit directory under work_dir for checkpoints and config
    exp_name='model=resnet50-dataset=cifar10-lr=1e-1',
    logger_interval=50,
    # monitor metric for saving checkpoints
    monitor='val_accuracy',
    # logger type, support TensorboardLogger, CometLogger
    logger=[dict(type='comet', key='Your Key Here!')],
    # checkpoint saving settings
    checkpoint=dict(type='ModelCheckpoint',
                    top_k=1,
                    mode='max',
                    verbose=True,
                    save_last=False,
                    ),
    # early stopping settings
    earlystopping=dict(
            mode='max',
            strict=False,
            patience=160,
            min_delta=0.0001,
            check_finite=True,
            verbose=True
    )

)

# yapf:enable
# resume from a checkpoint
resume_from = None
cudnn_benchmark = True

# optimization
optimization = dict(
    # running time unit, support epoch and iter
    type='epoch',
    # total running units
    max_iters=100,
    # optimizer
    optimizer=dict(type='SGD', lr=1e-1, weight_decay=0.0005, momentum=0.9),
    # learning rate scheduler and warmup
    scheduler=dict(type='OneCycle',
                   max_lr=1.2e-1,
                   interval='step',
                   # warmup=dict(type='LinearWarmup', period=0.1)
                   )

)
