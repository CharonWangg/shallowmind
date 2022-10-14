# model settings
# 10 classes for cifar10
num_classes = 128
# loss function, set multiple losses if needed, allow weighted sum
loss = [dict(type='TorchLoss', loss_name='CosineEmbeddingLoss', loss_weight=1.0)]
auxiliary_loss = [dict(type='HSICLoss', loss_weight=1.0)]
skip_val = True

patch_size = 16
model = dict(
    # Base Encoder Decoder Architecture (Backbone+Head)
    type='CausalVisualEncoder',
    # Visual Backbone from Timm Models library
    front_backbone=dict(
        type='TimmModels',
        model_name='resnet18',
        features_only=True,
        remove_fc=True,
        pretrained=False
    ),
    patch_embedding=dict(type='LinearEmbedding',
                           input_length=256,
                           in_channels=98,
                           embedding_size=128,
                           position=True),
    back_backbone=dict(
        type='Transformer',
        in_channels=128,
        hidden_size=128,
        num_layers=4,
        nhead=8,
    ),
    # Linear Head for classification
    head=dict(
        type='BaseHead',
        in_channels=128,
        in_index=-1,
        dropout=0.0,
        num_classes=num_classes,
        channels=None,
        losses=loss
    ),
    # auxiliary_head, only work in training for auxiliary loss
    auxiliary_head=dict(
        type='BaseHead',
        in_channels=128,
        in_index=-1,
        dropout=0.0,
        num_classes=num_classes,
        channels=None,
        losses=auxiliary_loss
    ),
    # metrics for evaluation, allow multiple metrics
    evaluation=dict(metrics=[dict(type='TorchMetrics', metric_name='Accuracy',
                                    prob=True)])

)

# dataset settings
dataset_type = 'DualImageNet'
dataset_name = 'CIFAR10'

# training data preprocess pipeline
train_pipeline = dict(orig_pipeline=[
                                    dict(type='LoadImages'),
                                    dict(type='Albumentations', transforms=[
                                        dict(type='RandomResizedCrop', height=224, width=224, scale=(0.2, 1.), p=1.0),
                                        dict(type='Normalize', mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225], p=1.0)]),
                                    dict(type='ToTensor')],
                      aug_pipeline=[dict(type='LoadImages'),
                                    dict(type='Albumentations', transforms=[
                                        dict(type='RandomResizedCrop', height=224, width=224, scale=(0.2, 1.), p=1.0),
                                        dict(type='ColorJitter', brightness=0.2, contrast=0.2, saturation=0.2,
                                                                 hue=0.2, p=0.8),
                                        dict(type='GaussianBlur', sigma_limit=[0.1, 2.0], p=0.5),
                                        dict(type='HorizontalFlip'),
                                        dict(type='Normalize', mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225], p=1.0)]),
                                    dict(type='ToTensor')]
)

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
        data_root='/data2/charon/imagenet',
        train=True,
        download=True,
        sampler=None,  # None is default sampler, set to RandomSampler/DistributedSampler
        pipeline=train_pipeline
        ),
    val=dict(
        type=dataset_type,
        dataset_name=dataset_name,
        data_root='/data2/charon/imagenet',
        train=False,
        sampler='SequentialSampler',
        pipeline=test_pipeline,
        ),
    test=dict(
        type=dataset_type,
        data_root='/data2/charon/imagenet',
        dataset_name=dataset_name,
        train=False,
        sampler='SequentialSampler',
        pipeline=test_pipeline
    ),
)

# yapf:disable
log = dict(
    # project name, used for cometml
    project_name='causal_discovery_in_image',
    # work directory, used for saving checkpoints and loggings
    work_dir='/data2/charon/test/ckpts',
    # explicit directory under work_dir for checkpoints and config
    exp_name='model=resnet18-dataset=imagenet-lr=1e-2',
    logger_interval=50,
    # monitor metric for saving checkpoints
    monitor='train_loss_epoch',
    # logger type, support TensorboardLogger, CometLogger
    logger=[dict(type='comet', key='Your Key Here!')],
    # checkpoint saving settings
    checkpoint=dict(type='ModelCheckpoint',
                    top_k=1,
                    mode='max',
                    verbose=True,
                    save_last=False,
                    ),
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
    optimizer=dict(type='SGD', lr=1e-2, weight_decay=0.0005, momentum=0.9),
    # learning rate scheduler and warmup
    scheduler=dict(type='CosineAnnealing',
                   interval='step',
                   # warmup=dict(type='LinearWarmup', period=0.1)
                   )

)
