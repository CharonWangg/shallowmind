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
    if cfg.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    # data
    data_module = DataInterface(cfg.data)

    # model
    # TODO: more elegantly infer the max_iters when in max_epochs mode
    # optimization
    if cfg.optimization.type == 'epoch':
        args.max_epochs = cfg.optimization.max_iters
        cfg.optimization.max_epochs = cfg.optimization.max_iters
        cfg.optimization.max_iters = data_module.inner_iters * cfg.optimization.max_epochs
    elif cfg.optimization.type == 'iter':
        args.max_steps = cfg.optimization.max_iters
        cfg.optimization.max_epochs = cfg.optimization.max_iters//data_module.inner_iters
    else:
        raise NotImplementedError('You must choose optimziation update step from (epoch/iter)')

    # for models need setting readout layer with dataloader information
    if cfg.model.pop('need_dataloader', False):
        cfg.model.dataloader = data_module.train_dataloader()

    if cfg.get('resume_from', None) is None:
        model = ModelInterface(cfg.model, cfg.optimization)
    else:
        model = ModelInterface(cfg.model, cfg.optimization).load_from_checkpoint(cfg.resume_from)

    # log
    # callbacks
    callbacks = []
    # used to control early stopping
    if cfg.log.earlystopping is not None:
        callbacks.append(plc.EarlyStopping(
            monitor=cfg.log.monitor,
            mode=cfg.log.earlystopping.mode,
            strict=cfg.log.earlystopping.strict,
            patience=cfg.log.earlystopping.patience,
            min_delta=cfg.log.earlystopping.min_delta,
            check_finite=cfg.log.earlystopping.check_finite,
            verbose=cfg.log.earlystopping.verbose
        ))
    # used to save the best model
    if cfg.log.checkpoint is not None:
        if cfg.log.checkpoint.type == 'ModelCheckpoint':
            callbacks.append(plc.ModelCheckpoint(
                monitor=cfg.log.monitor,
                dirpath=os.path.join(cfg.log.work_dir, cfg.log.exp_name, 'ckpts'),
                filename=f'exp_name={cfg.log.exp_name}-' + \
                        f'cfg={cfg.base_name.strip(".py")}-' + \
                        f'{{{cfg.log.monitor}:.3f}}',
                save_top_k=cfg.log.checkpoint.top_k,
                mode=cfg.log.checkpoint.mode,
                verbose=cfg.log.checkpoint.verbose,
                save_last=cfg.log.checkpoint.save_last
            ))
        else:
            raise NotImplementedError("Other kind of checkpoints haven't implemented!")

    if cfg.optimization.scheduler is not None:
        callbacks.append(plc.LearningRateMonitor())

    args.callbacks = callbacks

    # Disable ProgressBar
    # callbacks.append(plc.progress.TQDMProgressBar(
    #     refresh_rate=0,
    # ))

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

    if cfg.resume_from is None:
        trainer.fit(model, data_module)
    else:
        trainer.fit(model, data_module, ckpt_path=cfg.resume_from)

    trainer.test(model, data_module)

if __name__ == '__main__':
    train()


