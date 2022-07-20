from ..utils.config import Registry, build_from_cfg

DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')
SAMPLERS = Registry('sampler')

def build_pipeline(cfg):
    '''build pipeline with given config'''
    return build_from_cfg(cfg, PIPELINES)

def build_dataset(cfg):
    '''build dataset with given config'''
    return build_from_cfg(cfg, DATASETS)

def build_sampler(cfg):
    '''build sampler with given config'''
    return build_from_cfg(cfg, SAMPLERS)

