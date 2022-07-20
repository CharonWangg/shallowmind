import pytorch_lightning as pl
import torch.nn as nn
from ..builder import LOSSES

@LOSSES.register_module()
class TorchLoss(pl.LightningModule):
    def __init__(self, loss_name=None, step_reduction=False, loss_weight=1.0, **kwargs):
        super(TorchLoss, self).__init__()
        if loss_name is None:
            raise ValueError('loss_name is required')
        self.loss_name = loss_name
        self.loss_weight = loss_weight
        self.step_reduction = step_reduction
        self.loss = getattr(nn, self.loss_name)(**kwargs)

    def forward(self, pred, target):
        if self.step_reduction:
            pred = pred.mean(dim=1)

        return self.loss(pred, target)