import os
import sys
import ast
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
# ugly hack to enable configs inside the package to be run
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shallowmind.src.model import ModelInterface
from shallowmind.src.data import DataInterface
from argparse import ArgumentParser
from shallowmind.src.utils import load_config


torch.multiprocessing.set_sharing_strategy('file_system')


def search_cfg_ckpt(target_dir, keyword=None, screen=None, target_suffix=["ckpt", "py"]):
    # search files by given keywords and suffix, and screened keywords under target directory
    find_res = []
    target_suffix_dot = ["." + suffix for suffix in target_suffix]
    walk_generator = os.walk(target_dir)
    for root_path, dirs, files in walk_generator:
        if len(files) < 1:
            continue
        for file in files:
            file_name, suffix_name = os.path.splitext(file)
            if suffix_name in target_suffix_dot:
                file_name = os.path.join(root_path, file)
                # keyword check
                if keyword is not None:
                    _check = 0
                    for word in keyword:
                        if word in file_name:
                            _check += 1
                    if screen is not None:
                        for screen_word in screen:
                                if screen_word in file_name:
                                    _check -= 1
                    if _check == len(keyword):
                            find_res.append(file_name)
                else:
                    find_res.append(file_name)
    return find_res


def prepare_inference(cfg, ckpt):
    # load model and data config settings from config file
    cfg = load_config(cfg)
    # 5 part: model(arch, loss, ) -> ModelInterface /data(file i/o, preprocess pipeline) -> DataInterface
    # /optimization(optimizer, scheduler, epoch/iter...) -> ModelInterface/
    # log(logger, checkpoint, work_dir) -> Trainer /other

    # data
    data_module = DataInterface(cfg.data)

    # for models need setting readout layer with dataloader information
    if cfg.model.pop('need_dataloader', False):
        data_module.setup(stage='fit')
        cfg.model.dataloader = data_module.train_dataloader()

    # load checkpoint
    if os.path.exists(ckpt):
        model = ModelInterface.load_from_checkpoint(ckpt, model=cfg.model, optimization=cfg.optimization)
    else:
        raise FileNotFoundError(f'checkpoint file {ckpt} not found')

    return data_module, model

def infer():
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
        cfg.model.dataloader = data_module.train_dataloader()

    # load checkpoint
    if os.path.exists(args.ckpt):
        model = ModelInterface(cfg.model, cfg.optimization).load_from_checkpoint(args.ckpt)
    else:
        raise FileNotFoundError(f'checkpoint file {args.ckpt} not found')

    # Disable ProgressBar
    # callbacks.append(plc.progress.TQDMProgressBar(
    #     refresh_rate=0,
    # ))

    # load trainer
    trainer = Trainer.from_argparse_args(args)
    trainer.infer(model, data_module)


if __name__ == '__main__':
    infer()


