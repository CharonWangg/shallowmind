import torch.nn as nn
import pytorch_lightning as pl
from ..builder import build_loss, HEADS

@HEADS.register_module()
class BaseHead(pl.LightningModule):
    def __init__(self, in_channels=32, channels=None, num_classes=2, dropout=0.1, in_index=-1,
                 act_cfg=dict(type='ReLU'), norm_cfg=dict(type='BatchNorm1d'),
                 losses=dict(type='TorchLoss', loss_name='CrossEntropyLoss', loss_weight=1.0),):
        super(BaseHead, self).__init__()
        self.__dict__.update(locals())

        self.losses = nn.ModuleList()
        if isinstance(losses, dict):
            self.losses.append(build_loss(losses))
        elif isinstance(losses, list):
            for loss in losses:
                self.losses.append(build_loss(loss))
        else:
            raise TypeError(f'losses must be a dict or sequence of dict,\
                   but got {type(losses)}')

        self.dropout = (nn.Dropout(dropout) if dropout > 0 else None)
        self.norm = getattr(nn, norm_cfg['type'])
        self.activation = getattr(nn, act_cfg['type'])()
        # 3 layers MLP
        if channels is None:
            self.channels = [in_channels, num_classes]
            # last layer
            # self.model = nn.Sequential(*[nn.LayerNorm(in_channels),
            #                              self.dropout,
            #                              nn.Linear(*self.channels)])
            # experimentally, the last layer is better
            self.model = nn.Sequential(*[nn.Linear(*self.channels)])
        elif isinstance(channels, int):
            self.model = nn.Sequential(self.dropout,
                                       nn.Linear(in_channels, channels),
                                       self.norm(channels),
                                       self.activation,
                                       nn.Linear(channels, channels),
                                       self.norm(channels),
                                       self.activation,
                                       nn.Linear(channels, num_classes))
        elif isinstance(channels, list):
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

    def forward(self, x, **kwargs):
        '''use specific backbone layer output to forward'''
        return self.model(x[self.in_index], **kwargs)

    def forward_train(self, input, label, **kwargs):
        '''forward for training'''
        output = self.forward(input, **kwargs)
        losses = self.parse_losses(output, label)

        return losses

    def forward_test(self, input, label=None, **kwargs):
        '''forward for testing'''
        output = self.forward(input, **kwargs)
        if label is not None:
            losses = self.parse_losses(output, label)
            return {**{'output':output}, **losses}
        else:
            return {'output': output}

    def parse_losses(self, pred, label):
        """Compute loss"""
        loss = dict()
        # TODO: add sample weight to loss calculation
        for _loss in self.losses:
            if _loss.loss_name not in loss:
                loss[_loss.loss_name] = _loss(pred, label) * _loss.loss_weight
            else:
                loss[_loss.loss_name] += _loss(pred, label) * _loss.loss_weight

        return loss

