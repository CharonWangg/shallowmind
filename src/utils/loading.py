import os
from .config import Config

# load config from config file
def load_config(cfg_path=None):
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f'{cfg_path} not existed!')
    cfg = Config.fromfile(cfg_path)
    return cfg