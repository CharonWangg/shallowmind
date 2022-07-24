from ..utils.config import Registry, build_from_cfg

ARCHS = Registry('arch')
BACKBONES = Registry('backbone')
NECKS = Registry('neck')
HEADS = Registry('head')
OPTIMIZERS = Registry('optimizer')
SCHEDULERS = Registry('scheduler')
LOSSES = Registry('loss')
METRICS = Registry('metric')

def build_arch(cfg):
    '''build arch with given config'''
    return build_from_cfg(cfg, ARCHS)

def build_backbone(cfg):
    '''build backbone with given config'''
    return build_from_cfg(cfg, BACKBONES)

def build_neck(cfg):
    '''build neck with given config'''
    return build_from_cfg(cfg, NECKS)

def build_head(cfg):
    '''build head with given config'''
    return build_from_cfg(cfg, HEADS)

def build_optimizer(cfg):
    '''build optimizer with given config'''
    return build_from_cfg(cfg, OPTIMIZERS)

def build_scheduler(cfg):
    '''build scheduler with given config'''
    return build_from_cfg(cfg, SCHEDULERS)

def build_loss(cfg):
    '''build loss with given config'''
    return build_from_cfg(cfg, LOSSES)

def build_metric(cfg):
    '''build metric with given config'''
    return build_from_cfg(cfg, METRICS)






