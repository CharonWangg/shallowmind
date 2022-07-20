import os
from mmcv.utils import config

# load config from config file
def load_config(cfg_path=None):
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f'{cfg_path} not existed!')
    cfg = config.Config.fromfile(cfg_path)
    return cfg