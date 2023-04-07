import os
import ast
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import WandbLogger

# ugly hack to enable configs inside the package to be run
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)

# for ray
os.environ["TUNE_ORIG_WORKING_DIR"] = os.getcwd()
os.environ["PYTHONPATH"] = parent_dir + ":" + os.environ.get("PYTHONPATH", "")

from argparse import ArgumentParser
from src.utils import Config


# hyperparameter search
import ray
from ray import air, tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining

ray.init(ignore_reinit_error=True, num_cpus=8)
torch.multiprocessing.set_sharing_strategy("file_system")


def configure_arguments():
    """
    Configure the arguments from command line
    """
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument("--cfg", type=str, help="config file path")
    parser.add_argument("--hp_cfg", type=str, help="config file path for search")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--gpu_ids", default="0", type=str)
    parser.add_argument('--work_dir', type=str, default='')
    args = parser.parse_args()
    return args


def run(config):
    os.chdir(os.environ["TUNE_ORIG_WORKING_DIR"])

    args, hp_config = config["args"], config["hp_config"]
    # device setting
    cfg = Config.fromfile(args.cfg)
    # fix random seed (seed in config has higher priority )
    seed = cfg.seed if cfg.get('seed', None) is not None else args.seed
    # command line arguments have higher priority
    cfg.seed = seed
    args.accelerator = 'auto'
    args.devices = ast.literal_eval(args.gpu_ids)
    if args.work_dir:
        cfg.work_dir = args.work_dir

    # ****************************** Set Hyper Parameters from Arguments ******************************** #
    # *************************************************************************************************** #

    # fix random seed (seed in config has higher priority )
    pl.seed_everything(seed)

    # ***************************************** Model Module ******************************************** #
    # model =
    # *************************************************************************************************** #

    # ***************************************** Data Module ********************************************* #
    # data_module =
    # *************************************************************************************************** #

    # **************************************** Optimization ********************************************** #
    args.max_epochs = cfg.max_epochs
    # **************************************************************************************************** #

    # ****************************************** Callbacks *********************************************** #
    callbacks = []
    callbacks.append(plc.progress.TQDMProgressBar(refresh_rate=0))
    callbacks.append(plc.EarlyStopping(**cfg.early_stopping))
    callbacks.append(TuneReportCheckpointCallback(metrics=hp_config["monitor"].metrics, on="validation_end"))
    args.callbacks = callbacks
    # **************************************************************************************************** #

    # ****************************************** Logger ************************************************** #
    save_dir = os.path.join(cfg.work_dir, cfg.exp_name, 'log')
    os.makedirs(save_dir, exist_ok=True)
    if cfg.get('logging', True):
        args.logger = [WandbLogger(name=cfg.exp_name, project=cfg.project_name, save_dir=save_dir)]
    # **************************************************************************************************** #

    # ****************************************** Trainer ************************************************* #
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, data_module, ckpt_path=cfg.get("resume_from", None))


def sweep():
    args = configure_arguments()
    hp_config = Config.fromfile(args.hp_cfg)
    exp_name = hp_config.exp_name
    n_trials = hp_config.n_trials
    cpu_limit = hp_config.get("cpu", 4)
    gpu_limit = hp_config.get("gpu", 1)
    metric = hp_config.monitor

    hp_config = {k: getattr(tune, v["type"])(v["range"]) for k, v in hp_config.hyper_parameters.items()}
    hp_config["monitor"] = metric

    scheduler = ASHAScheduler(
        max_t=Config.fromfile(args.cfg).optimization.max_iters,
        grace_period=1,
        reduction_factor=2)

    tuner = tune.Tuner(
        tune.with_resources(
            run,
            resources={
            "cpu": cpu_limit,
            "gpu": gpu_limit
            }
        ),
        tune_config=tune.TuneConfig(
            metric=hp_config["monitor"].target,
            mode=hp_config["monitor"].direction,
            scheduler=scheduler,
            num_samples=n_trials,
        ),
        run_config=air.RunConfig(
            name=exp_name,
        ),
        param_space={"args": args,
                     "hp_config": hp_config},
    )
    analysis = tuner.fit()

    print("best result:", analysis.get_best_result())


if __name__ == "__main__":
    sweep()
