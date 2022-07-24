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

def test():
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--cfg', type=str, help='config file path')
    parser.add_argument('--ckpt', type=str, help='checkpoint file path')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu_ids', default='0', type=str)

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
    # 5 part: model(arch, loss, ) -> ModelInterface /data(file i/o, preprocess pipeline) -> DataInterface
    # /optimization(optimizer, scheduler, epoch/iter...) -> ModelInterface/
    # log(logger, checkpoint, work_dir) -> Trainer /other

   # other setting
    if cfg.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    # data
    data_module = DataInterface(cfg.data)

    # for models need setting readout layer with dataloader information
    if cfg.model.pop('need_dataloader', False):
        data_module.setup(stage='fit')
        cfg.model.dataloader = data_module.train_dataloader()

    # load checkpoint
    if os.path.exists(args.ckpt):
        model = ModelInterface.load_from_checkpoint(args.ckpt,
                                                    model=cfg.model,
                                                    optimization=cfg.optimization)
    else:
        raise FileNotFoundError(f'checkpoint file {args.ckpt} not found')

    # Disable ProgressBar
    # callbacks.append(plc.progress.TQDMProgressBar(
    #     refresh_rate=0,
    # ))

    # load trainer
    trainer = Trainer.from_argparse_args(args)
    trainer.test(model, data_module)

if __name__ == '__main__':
    test()


