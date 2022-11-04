import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from ..builder import BACKBONES, build_embedding
from ..utils import ResidualAdd


class MLPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_cfg=dict(type='ReLU'), norm_cfg=dict(type='BatchNorm1d'),
                 dropout=0.0, residual=False, **kwargs):
        super(MLPBlock, self).__init__()
        self.block = nn.Sequential(nn.Linear(in_channels, out_channels, **kwargs),
                                   getattr(nn, norm_cfg.get('type'))(out_channels,
                                                                     **{k: v for k, v in norm_cfg.items()
                                                                        if k != 'type'}),
                                   getattr(nn, act_cfg.get('type'))(**{k: v for k, v in act_cfg.items()
                                                                       if k != 'type'}),
                                   nn.Dropout(dropout)
                                   )
        if residual:
            self.block = ResidualAdd(self.block)

    def forward(self, x):
        return self.block(x)


@BACKBONES.register_module()
class MLP(pl.LightningModule):
    def __init__(self, in_channels=32, hidden_size=[64, 128], dropout=0.0,
                 act_cfg=dict(type='ReLU'), norm_cfg=dict(type='BatchNorm1d'),
                 residual=False, embedding=None, **kwargs):
        self.__dict__.update(locals())
        super().__init__(**kwargs)

        if embedding is not None:
            self.embedding = build_embedding(embedding)
            if in_channels is None:
                in_channels = self.embedding.embedding_size
        else:
            self.embedding = None

        self.model = nn.Sequential(*[MLPBlock(in_channels, hidden_size[0],
                                              dropout=dropout, residual=residual,
                                              act_cfg=act_cfg, norm_cfg=norm_cfg, **kwargs),
                                     *[MLPBlock(hidden_size[i], hidden_size[i + 1],
                                                dropout=dropout, residual=residual,
                                                act_cfg=act_cfg, norm_cfg=norm_cfg, **kwargs)
                                       for i in range(len(hidden_size) - 1)]
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
