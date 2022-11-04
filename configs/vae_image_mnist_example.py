# model settings
# 10 classes for mnist
num_classes = 10
# loss function, set multiple losses if needed, allow weighted sum
reg_loss = dict(type='KLDivergenceLoss', loss_weight=0.1)
rec_loss = dict(type='TorchLoss', loss_name='MSELoss', step_reduction=None, loss_weight=1.0)

model = dict(
    # Base VAE Architecture (recognition model + generative model)
    type='BaseVAE',
    need_dataloader=True,
    encoder=dict(
        backbone=dict(
            type='BaseConvNet',
            in_channels=1,
            hidden_size=[32, 64],
            kernel_size=[5, 5],
            stride=[1, 1],
            padding=[0, 0],
            act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
        ),
        head=dict(
            type='BaseHead',
            in_index=-1,
            dropout=0.0,
            num_classes=128,
            losses=reg_loss
        ),
    ),
    decoder=dict(
        type='ConvTransHead',
        in_channels=128,
        hidden_size=[64, 1],
        kernel_size=[4, 4],
        stride=[2, 2],
        padding=[1, 1],
        act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
        losses=rec_loss,
    ),
    evaluation=dict(metrics=[dict(type='ImageVisualization', save=True, n_samples=3)])
)

# dataset settings
dataset_type = 'TorchVision'
dataset_name = 'MNIST'
data_root = '.cache'

# training data preprocess pipeline
train_pipeline = [dict(type='LoadImages', to_RGB=False),
                  dict(type='ToTensor')]

# validation data preprocess pipeline
test_pipeline = [dict(type='LoadImages', to_RGB=False),
                 dict(type='ToTensor')]

data = dict(
    train_batch_size=256,  # for single card
    val_batch_size=256,
    test_batch_size=256,
    num_workers=4,
    train=dict(
        type=dataset_type,
        dataset_name=dataset_name,
        data_root=data_root,
        train=True,
        sampler=None,  # None is default sampler, set to RandomSampler/DistributedSampler
        pipeline=train_pipeline
        ),
    val=dict(
        type=dataset_type,
        dataset_name=dataset_name,
        data_root=data_root,
        train=False,
        sampler='SequentialSampler',
        pipeline=test_pipeline,
        ),
    test=dict(
        type=dataset_type,
        dataset_name=dataset_name,
        data_root=data_root,
        train=False,
        sampler='SequentialSampler',
        pipeline=test_pipeline
    ),
)

# yapf:disable
log = dict(
    # project name, used for cometml
    project_name='vae_test',
    # work directory, used for saving checkpoints and loggings
    work_dir='work_dir',
    # explicit directory under work_dir for checkpoints and config
    exp_name='model=vae-dataset=mnist-lr=1e-4',
    logger_interval=50,
    # monitor metric for saving checkpoints
    monitor='step',
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
    earlystopping=None

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
    optimizer=dict(type='Adam', lr=1e-4),
    # learning rate scheduler and warmup
    scheduler=None
)
