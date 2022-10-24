import torch.nn as nn
import pytorch_lightning as pl
from ..builder import build_loss, HEADS
from .base import BaseLayer


@HEADS.register_module()
class BaseHead(BaseLayer):
    def __init__(self, in_channels=32, channels=None, num_classes=2, dropout=0.1, in_index=-1,
                 act_cfg=dict(type='ReLU'), norm_cfg=dict(type='BatchNorm1d'),
                 losses=dict(type='TorchLoss', loss_name='CrossEntropyLoss', loss_weight=1.0),
                 **kwargs):
        self.__dict__.update(locals())
        super(BaseHead, self).__init__(losses=losses, **kwargs)
        self.dropout = (nn.Dropout(dropout) if dropout > 0 else None)
        self.norm = getattr(nn, norm_cfg['type'])
        self.activation = getattr(nn, act_cfg['type'])()
        if channels is None:
            # one linear layer
            self.channels = [in_channels, num_classes]
            # last layer
            # self.model = nn.Sequential(*[nn.LayerNorm(in_channels),
            #                              self.dropout,
            #                              nn.Linear(*self.channels)])
            # experimentally, the last layer is better
            self.model = nn.Sequential(*[nn.Linear(*self.channels)])
        elif isinstance(channels, int):
            # 3 layers MLP
            self.model = nn.Sequential(self.dropout,
                                       nn.Linear(in_channels, channels),
                                       self.norm(channels),
                                       self.activation,
                                       nn.Linear(channels, channels),
                                       self.norm(channels),
                                       self.activation,
                                       nn.Linear(channels, num_classes))
        elif isinstance(channels, list):
            # multi layers MLP
            channels = [in_channels] + channels + [num_classes]
            self.model = []
            for depth in range(len(channels) - 2):
                self.model.extend([self.dropout,
                                   nn.Linear(channels[depth], channels[depth + 1]),
                                   self.norm(channels[depth + 1]),
                                   self.activation])
            self.model = nn.Sequential(*self.model)
            self.model.append(nn.Linear(channels[-2], channels[-1]))
        else:
            raise TypeError(f'channels must be an int or sequence of int')