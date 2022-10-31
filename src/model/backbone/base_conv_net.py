import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from ..builder import BACKBONES, build_embedding
from ..utils import ResidualAdd


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                    act_cfg=dict(type='ReLU'), norm_cfg=dict(type='BatchNorm2d'), dropout=0.0,
                    residual=False, **kwargs):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, **kwargs),
                                   getattr(nn, norm_cfg.get('type'))(out_channels, **{k: v for k, v in norm_cfg.items()
                                                                                      if k != 'type'}),
                                   getattr(nn, act_cfg.get('type'))(**{k: v for k, v in act_cfg.items()
                                                                       if k != 'type'}),
                                   nn.Dropout(dropout)
                                   )
        if residual:
            self.block = ResidualAdd(self.block)
        self._init_weights()

    def _init_weights(self):
        # init weights with kaiming_normal_:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.block(x)


@BACKBONES.register_module()
class BaseConvNet(pl.LightningModule):
    def __init__(self, in_channels=3, hidden_size=[16, 32, 64, 128, 256, 512], kernel_size=[3, 3, 3, 3, 3, 3],
                 stride=[2, 1, 2, 1, 2, 1], padding=[1, 1, 1, 1, 1, 1], dropout=0.0,
                 act_cfg=dict(type='ReLU'), norm_cfg=dict(type='BatchNorm2d'),
                 residual=False, embedding=None, **kwargs):
        self.__dict__.update(locals())
        super().__init__(**kwargs)

        if embedding is not None:
            self.embedding = build_embedding(embedding)
            if in_channels is None:
                in_channels = self.embedding.embedding_size
        else:
            self.embedding = None

        self.model = nn.Sequential(*[ConvBlock(in_channels, hidden_size[0], kernel_size[0], stride[0], padding[0],
                                               dropout=dropout, residual=residual,
                                               act_cfg=act_cfg, norm_cfg=norm_cfg, **kwargs),
                                     *[ConvBlock(hidden_size[i], hidden_size[i+1], kernel_size[i+1],
                                                 stride[i+1], padding[i+1], dropout=dropout, residual=residual,
                                                 act_cfg=act_cfg, norm_cfg=norm_cfg, **kwargs)
                                       for i in range(len(hidden_size)-1)]
                                     ])

    def forward(self, x):
        if isinstance(x, dict):
            assert 'image' or 'seq' in x.keys(), 'input must be a dict with key "image" or "seq"'
            x = x['image'] if 'image' in x.keys() else x['seq']
        # high level projection:
        if self.embedding is not None:
            x = self.embedding(x)
        x = self.model(x)
        return [x]