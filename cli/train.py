import os
import ast
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning.callbacks as plc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from argparse import ArgumentParser
from src.utils import Config

torch.multiprocessing.set_sharing_strategy('file_system')


def setup():
    """
    Setup command line arguments and load config file
    """
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--cfg', type=str, help='config file path')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu_ids', default='[0]', type=str)
    parser.add_argument('--work_dir', type=str, default='')
    args = parser.parse_args()

    cfg = Config.fromfile(args.cfg)
    # fix random seed (seed in config has higher priority )
    seed = cfg.seed if cfg.get('seed', None) is not None else args.seed
    # command line arguments have higher priority
    cfg.seed = seed
    args.accelerator = 'auto'
    args.devices = ast.literal_eval(args.gpu_ids)
    if args.work_dir:
        cfg.work_dir = args.work_dir

    # reproducibility
    pl.seed_everything(seed)
    return args, cfg


def train():
    """
    Main training function
    """
    args, cfg = setup()

    # ***************************************** Model Module ******************************************** #
    # model =
    # *************************************************************************************************** #

    # ***************************************** Data Module ********************************************* #
    # data_module =
    # *************************************************************************************************** #

    # **************************************** Optimization ********************************************** #
    args.max_epochs = cfg.max_epochs
    # **************************************************************************************************** #

    # ****************************************** Logging ************************************************** #
    # save config file to log directory
    cfg_name = args.cfg.split('/')[-1]
    if os.path.exists(os.path.join(cfg.work_dir, cfg.exp_name)):
        cfg.dump(os.path.join(cfg.work_dir, cfg.exp_name, cfg_name))
    else:
        os.makedirs(os.path.join(cfg.work_dir, cfg.exp_name))
        cfg.dump(os.path.join(cfg.work_dir, cfg.exp_name, cfg_name))

    # logger
    save_dir = os.path.join(cfg.work_dir, cfg.exp_name, 'log')
    os.makedirs(save_dir, exist_ok=True)
    if cfg.get('logging', True):
        args.logger = [WandbLogger(name=cfg.exp_name, project=cfg.project_name, save_dir=save_dir)]
    # **************************************************************************************************** #

    # ****************************************** Callbacks *********************************************** #
    callbacks = [plc.RichProgressBar(), plc.EarlyStopping(**cfg.early_stopping)]
    # used to control early stopping
    # used to save the best model
    dirpath = os.path.join(cfg.work_dir, cfg.exp_name, 'ckpts')
    os.makedirs(dirpath, exist_ok=True)
    filename = f'exp_name={cfg.exp_name}-' + f'{{{cfg.early_stopping.monitor}:.4f}}'
    callbacks.append(plc.ModelCheckpoint(dirpath=dirpath, filename=filename, **cfg.checkpoint))
    args.callbacks = callbacks
    # **************************************************************************************************** #

    # ****************************************** Trainer ************************************************** #
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, data_module)
    # **************************************************************************************************** #


if __name__ == '__main__':
    train()


