import torch.nn as nn
import pytorch_lightning as pl
from ..builder import build_loss, HEADS


@HEADS.register_module()
class BaseLayer(pl.LightningModule):
    def __init__(self, losses=None, **kwargs):
        super(BaseLayer, self).__init__(**kwargs)
        if losses is not None:
            self.losses = nn.ModuleList()
            if isinstance(losses, dict):
                self.losses.append(build_loss(losses))
            elif isinstance(losses, list):
                for loss in losses:
                    self.losses.append(build_loss(loss))
            else:
                raise TypeError(f'losses must be a dict or sequence of dict,\
                       but got {type(losses)}')
        else:
            self.losses = None

    def forward(self, x, **kwargs):
        '''use specific backbone layer output to forward'''
        if isinstance(x, dict):
            x = x.pop('x')
        return self.model(x[self.in_index].view(x[self.in_index].shape[0], -1), **kwargs)

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

