import torch.nn as nn
import pytorch_lightning as pl
from ..builder import HEADS
from .base import BaseLayer


class ConvTransBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=1, padding=0,
                    act_cfg=dict(type='ReLU'), norm_cfg=dict(type='BatchNorm2d'), **kwargs):
        super(ConvTransBlock, self).__init__()
        self.block = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                                      stride, padding, **kwargs),
                                   getattr(nn, norm_cfg.get('type'))(out_channels, **{k: v for k, v in norm_cfg.items()
                                                                                      if k != 'type'}),
                                   getattr(nn, act_cfg.get('type'))(**{k: v for k, v in act_cfg.items()
                                                                            if k != 'type'}),
                                   )
        self._init_weights()

    def _init_weights(self):
        # init weights with kaiming_normal_:
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.block(x)


@HEADS.register_module()
class ConvTransHead(BaseLayer):
    def __init__(self, in_channels=384, hidden_size=[192, 96, 32, 3], kernel_size=[4, 4, 4, 4],
                 stride=[1, 2, 2, 2], padding=[0, 1, 1, 1], in_index=-1,
                 act_cfg=dict(type='ReLU'), norm_cfg=dict(type='BatchNorm2d'),
                 losses=dict(type='TorchLoss', loss_name='CrossEntropyLoss', loss_weight=1.0), **kwargs):
        self.__dict__.update(locals())
        super().__init__(losses=losses, **kwargs)
        self.model = nn.Sequential(*[ConvTransBlock(in_channels, hidden_size[0], kernel_size[0],
                                                    stride[0], padding[0], act_cfg=act_cfg, norm_cfg=norm_cfg, **kwargs),
                                     *[ConvTransBlock(hidden_size[i], hidden_size[i+1], kernel_size[i+1],
                                                      stride[i+1], padding[i+1], act_cfg=act_cfg, norm_cfg=norm_cfg, **kwargs)
                                       for i in range(len(hidden_size)-1)],
                                     nn.Sigmoid()])

    def forward(self, x, **kwargs):
        x = self.model(x[self.in_index], **kwargs)
        return x
