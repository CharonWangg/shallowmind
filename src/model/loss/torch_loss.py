import pytorch_lightning as pl
import torch.nn as nn
from ..builder import LOSSES

@LOSSES.register_module()
class TorchLoss(pl.LightningModule):
    def __init__(self, loss_name=None, step_reduction=False, inverse=False, loss_weight=1.0, **kwargs):
        super(TorchLoss, self).__init__()
        if loss_name is None:
            raise ValueError('loss_name is required')
        self.loss_name = loss_name
        self.loss_weight = loss_weight
        # only work when dim is larger than 2
        self.step_reduction = step_reduction
        self.inverse = inverse
        self.loss = getattr(nn, self.loss_name)(**kwargs)

    def forward(self, pred, target):
        if pred.dim() > 2:
            if self.step_reduction:
                pred = pred.mean(dim=1)
            else:
                pred = pred.view(-1, pred.size(-1))
                target = target.view(-1)

        return self.loss(pred, target) if not self.inverse else -1 * self.loss(pred, target)