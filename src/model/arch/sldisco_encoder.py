import copy
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from ..utils import add_prefix
from ..builder import ARCHS
from .base import BaseArch
from ..head import BaseHead
from einops.layers.torch import Rearrange, Reduce


@ARCHS.register_module()
class SLDiscoEncoder(BaseArch):
    def __init__(self, backbone, head, **kwargs):
        super(SLDiscoEncoder, self).__init__(**kwargs)
        self.name = 'SLDiscoEncoder'
        assert backbone is not None, 'backbone is not defined'
        assert head is not None, 'head is not defined'

        self.input_length = backbone.get('input_length', None)
        assert self.input_length is not None, 'input_length is not defined'
        # column-wise, row-wise, entry-wise, local spatial
        self.backbone = nn.ModuleDict({'column': nn.Sequential(*[nn.Conv2d(1, 2**5, (self.input_length, 1)),
                                                                 Reduce('b c h w -> b c (repeat h) w',
                                                                        reduction='repeat',
                                                                        repeat=self.input_length),
                                                                 nn.ReLU()]),
                                       'row': nn.Sequential(*[nn.Conv2d(1, 2**5, (1, self.input_length)),
                                                              Reduce('b c h w -> b c h (repeat w)',
                                                                     reduction='repeat',
                                                                     repeat=self.input_length),
                                                              nn.ReLU()]),
                                       'entry': nn.Sequential(*[nn.Conv2d(1, 2**5, 1),
                                                                nn.ReLU()]),
                                       'local': nn.Sequential(*[nn.Conv2d(1, 2**5, 3,
                                                                          padding=1),
                                                                nn.ReLU()])})
        self.head = BaseHead(losses=head.losses)
        self.auxiliary_head = None
        # get the output shape of backbone
        with torch.no_grad():
            input = torch.zeros(1, 1, self.input_length, self.input_length, device=self.device)
            output = self.exact_feat(input)
        output_shape = output[0].flatten().shape[0]
        channels = head.get('channels', 128)
        self.head.model = nn.Sequential(*[nn.Dropout(0.2),
                                          nn.Linear(output_shape, channels),
                                          nn.ReLU(),
                                          nn.Dropout(0.2),
                                          nn.Linear(channels, self.input_length**2),
                                          nn.Flatten()])

    def exact_feat(self, x):
        if isinstance(x, dict):
            x = x['seq']
        x = torch.concat([v(x) for k, v in self.backbone.items()], dim=1)
        x = nn.MaxPool2d(2, stride=1)(x)
        x = x.view(x.shape[0], -1)
        return [x]