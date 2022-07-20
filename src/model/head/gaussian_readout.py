import torch.nn as nn
import pytorch_lightning as pl
from ..builder import build_loss, HEADS

@HEADS.register_module()
class GaussianReadout(pl.LightningModule):
    def __init__(self, in_channels, channels, num_classes, dropout_ratio=0.1,
                 act_cfg=dict(type='ReLU'), in_index=-1,
                 losses=dict(type='TorchLoss', loss_name='CrossEntropyLoss', loss_weight=1.0)):
        super(GaussianReadout, self).__init__()
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

        self.dropout = (nn.Dropout(dropout_ratio) if dropout_ratio > 0 else None)
        self.activation = getattr(nn, act_cfg['type'])()
        # 3 layers MLP
        self.model = nn.Sequential(self.dropout,
                                   nn.Linear(in_channels, channels),
                                   nn.BatchNorm1d(channels),
                                   self.activation,
                                   nn.Linear(channels, channels),
                                   nn.BatchNorm1d(channels),
                                   self.activation,
                                   nn.Linear(channels, num_classes))

    def forward(self, x):
        '''use specific backbone layer output to forward'''
        return self.model(x[self.in_index])

    def forward_train(self, input, label):
        '''forward for training'''
        output = self.forward(input)
        losses = self.parse_losses(output, label)

        return losses

    def forward_test(self, input, label=None):
        '''forward for testing'''
        output = self.forward(input)
        if label is not None:
            losses = self.parse_losses(output, label)
            return {'output':output}.update(losses)
        else:
            return {'output': output}

    def parse_losses(self, pred, label):
        """Compute segmentation loss."""
        loss = dict()
        # TODO: add sample weight to loss calculation
        for _loss in self.losses:
            if _loss.loss_name not in loss:
                loss[_loss.loss_name] = _loss(pred, label) * _loss.loss_weight
            else:
                loss[_loss.loss_name] += _loss(pred, label) * _loss.loss_weight


        return loss

