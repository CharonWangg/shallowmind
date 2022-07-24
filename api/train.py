import os
import ast
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, CometLogger
import pytorch_lightning.callbacks as plc
from shallowmind.src.model import ModelInterface
from shallowmind.src.data import DataInterface
from argparse import ArgumentParser
from shallowmind.src.utils import load_config

torch.multiprocessing.set_sharing_strategy('file_system')

def train():
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--cfg', type=str, help='config file path')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu_ids', default='0', type=str)
    # parser.add_argument('--accelerator', type=str, default="auto")
    # parser.add_argument('--precision', type=int, default=32, help='training precision (32/16)')
    # parser.add_argument('--limit_train_batches', type=float, default=1.0)
    # parser.add_argument('--limit_val_batches', type=float, default=1.0)
    # parser.add_argument('--limit_test_batches', type=float, default=1.0)
    # parser.add_argument('--gpus', type=int, nargs='+', help='ids of gpus to use')
    # parser.add_argument('--deterministic', action='store_true')

    args = parser.parse_args()

    # fix random seed
    pl.seed_everything(args.seed)

    # device setting
    args.accelerator = 'auto'
    # training setting for distributed training
    # args.gpu = '[0, 1, 2, 3]'
    args.gpus = ast.literal_eval(args.gpu_ids)
    args.devices = len(args.gpus)
    if args.devices > 1:
        args.sync_batchnorm = True
        args.strategy = 'ddp'


    # load model and data config settings from config file
    cfg = load_config(args.cfg)
    # save config file to log directory
    cfg.base_name = args.cfg.split('/')[-1]
    if os.path.exists(os.path.join(cfg.log.work_dir, cfg.log.exp_name)):
        cfg.dump(os.path.join(cfg.log.work_dir, cfg.log.exp_name, cfg.base_name))
    else:
        os.makedirs(os.path.join(cfg.log.work_dir, cfg.log.exp_name))
        cfg.dump(os.path.join(cfg.log.work_dir, cfg.log.exp_name, cfg.base_name))

    # 5 part: model(arch, loss, ) -> ModelInterface /data(file i/o, preprocess pipeline) -> DataInterface
    # /optimization(optimizer, scheduler, epoch/iter...) -> ModelInterface/
    # log(logger, checkpoint, work_dir) -> Trainer /other

   # other setting
    if cfg.get('cudnn_benchmark', True):
        torch.backends.cudnn.benchmark = True

    if cfg.get('deterministic', False):
        if cfg.get('cudnn_benchmark', True):
            print('cudnn_benchmark will be disabled')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # data
    data_module = DataInterface(cfg.data)

    # optimization
    if cfg.optimization.type == 'epoch':
        args.max_epochs = cfg.optimization.max_iters
    elif cfg.optimization.type == 'iter':
        args.max_steps = cfg.optimization.max_iters
    else:
        raise NotImplementedError('You must choose optimziation update step from (epoch/iter)')

    # for models need setting readout layer with dataloader information
    if cfg.model.pop('need_dataloader', False):
        data_module.setup(stage='fit')
        cfg.model.dataloader = data_module.train_dataloader()

    if cfg.get('resume_from', None) is None:
        model = ModelInterface(cfg.model, cfg.optimization)
    else:
        if not cfg.model.pop('pretrained', False):
            model = ModelInterface.load_from_checkpoint(cfg.resume_from,
                                                        model=cfg.model,
                                                        optimization=cfg.optimization)
        else:
            # load partial pretrained weights
            model = ModelInterface(cfg.model, cfg.optimization)
            pretrained_dict = torch.load(cfg.resume_from)['state_dict']
            model_dict = model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            print(f'Loaded pretrained state_dict: {pretrained_dict}')
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            # prevent optimization conflicting in the fresh training
            cfg.resume_from = None

    # log
    # callbacks
    callbacks = []
    # used to control early stopping
    if cfg.log.earlystopping is not None:
        callbacks.append(plc.EarlyStopping(
            monitor=cfg.log.get('monitor', 'val_loss'),
            mode=cfg.log.earlystopping.get('mode', 'max'),
            strict=cfg.log.earlystopping.get('strict', False),
            patience=cfg.log.earlystopping.get('patience', 5),
            min_delta=cfg.log.earlystopping.get('min_delta', 1e-5),
            check_finite=cfg.log.earlystopping.get('check_finite', True),
            verbose=cfg.log.earlystopping.get('verbose', True)
        ))
    # used to save the best model
    if cfg.log.checkpoint is not None:
        if cfg.log.checkpoint.type == 'ModelCheckpoint':
            callbacks.append(plc.ModelCheckpoint(
                monitor=cfg.log.monitor,
                dirpath=os.path.join(cfg.log.work_dir, cfg.log.exp_name, 'ckpts'),
                filename=f'exp_name={cfg.log.exp_name}-' + \
                        f'cfg={cfg.base_name.strip(".py")}-' + \
                        f'bs={cfg.data.train_batch_size}-'+ \
                        f'{{{cfg.log.monitor}:.3f}}',
                save_top_k=cfg.log.checkpoint.top_k,
                mode=cfg.log.checkpoint.mode,
                verbose=cfg.log.checkpoint.verbose,
                save_last=cfg.log.checkpoint.save_last
            ))
        else:
            raise NotImplementedError("Other kind of checkpoints haven't implemented!")

    # if cfg.optimization.scheduler is not None: (has been build in model_interface)
    #     callbacks.append(plc.LearningRateMonitor(logging_interval='step'))

    # Disable ProgressBar
    # callbacks.append(plc.progress.TQDMProgressBar(
    #     refresh_rate=0,
    # ))

    args.callbacks = callbacks

    # logger
    if cfg.log.logger is not None:
        loggers = []
        save_dir = os.path.join(cfg.log.work_dir, cfg.log.exp_name, 'log')
        args.log_every_n_steps = cfg.log.logger_interval
        for logger in cfg.log.logger:
            if 'comet' in logger.type:
                loggers.append(CometLogger(api_key=logger.key,
                                           save_dir=save_dir,
                                           project_name=cfg.log.project_name,
                                           rest_api_key=os.environ.get("COMET_REST_API_KEY"),
                                           experiment_key=os.environ.get("COMET_EXPERIMENT_KEY"),
                                           experiment_name=cfg.log.exp_name,
                                           display_summary_level=0))
            if 'tensorboard' in logger.type:
                loggers.append(TensorBoardLogger(save_dir=save_dir))
        args.logger = loggers

    # load trainer
    trainer = Trainer.from_argparse_args(args)

    trainer.fit(model, data_module, ckpt_path=cfg.get('resume_from', None))

    trainer.test(model, data_module)

if __name__ == '__main__':
    train()


