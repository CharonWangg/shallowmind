# model settings
# loss function, set multiple losses if needed, allow weighted sum
loss = dict(type='TorchLoss', loss_name='MSELoss', loss_weight=1.0)

model = dict(
    # Base GAN Architecture (Generator+Discriminator)
    type='BaseAgent',
    need_dataloader=True,
    # need to change the input conv layer to (kernel_size=3, stride=1, padding=1) to accept 32x32 input
    agent=dict(
        type='BaseEncoderDecoder',
        backbone=dict(
            type='MLP',
            hidden_size=[128],
            act_cfg=dict(type='ReLU'),
        ),
        head=dict(
            type='BaseHead',
            in_channels=128,
            in_index=-1,
            dropout=0.0,
            losses=loss
        ),
    ),
)

# dataset settings
dataset_type = 'Gym'
dataset_name = 'CartPole-v0'
data_root = '.cache'


data = dict(
    train_batch_size=16,  # for single card
    val_batch_size=16,  # for single card
    test_batch_size=16,  # for single card
    num_workers=1,
    train=dict(
        type=dataset_type,
        env_name=dataset_name,
        buffer_cfg=dict(buffer_size=1000),
        sample_size=200
        ),
    val=dict(
        type=dataset_type,
        env_name=dataset_name,
        buffer_cfg=dict(buffer_size=100),
        sample_size=10
        ),
    test=dict(
        type=dataset_type,
        env_name=dataset_name,
        buffer_cfg=dict(buffer_size=100),
        sample_size=10
    ),
)

# yapf:disable
log = dict(
    # project name, used for cometml
    project_name='dqn_test',
    # work directory, used for saving checkpoints and loggings
    work_dir='work_dir',
    # explicit directory under work_dir for checkpoints and config
    exp_name='model=dqn-dataset=cartepole-lr=1e-2',
    logger_interval=50,
    # monitor metric for saving checkpoints
    monitor='train_loss',
    # logger type, support TensorboardLogger, CometLogger
    logger=[dict(type='comet', key='oN1q8cGSIrH0zhorxKpNoenyc')],
    # checkpoint saving settings
    checkpoint=dict(type='ModelCheckpoint',
                    top_k=1,
                    mode='min',
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
    max_iters=150,
    # optimizer
    optimizer=dict(type='Adam', lr=1e-2),
    # learning rate scheduler and warmup
    scheduler=None
)
