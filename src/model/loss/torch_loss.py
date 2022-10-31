import pytorch_lightning as pl
import torch
import torch.nn as nn
from ..builder import LOSSES

@LOSSES.register_module()
class TorchLoss(pl.LightningModule):
    def __init__(self, loss_name=None, multi_label=False, step_reduction='flatten', inverse=False, loss_weight=1.0, to_float=False, **kwargs):
        super(TorchLoss, self).__init__()
        if loss_name is None:
            raise ValueError('loss_name is required')
        self.loss_name = loss_name
        self.loss_weight = loss_weight
        # only work when dim is larger than 2
        assert step_reduction in ['mean', 'flatten', None]
        self.step_reduction = step_reduction
        self.inverse = inverse
        self.multi_label = multi_label
        self.to_float = to_float
        self.loss = getattr(nn, self.loss_name)(**kwargs)

    def forward(self, pred, target):
        if pred.dim() > 2:
            if self.step_reduction == 'mean':
                pred = pred.mean(dim=1)
            elif self.step_reduction == 'flatten':
                pred = pred.view(-1, pred.size(-1))
                target = target.view(-1)
            elif self.step_reduction is None:
                pass
        else:
            if self.multi_label:
                pred = pred.view(-1)
                target = target.view(-1)

        if self.to_float:
            target = target.to(torch.float32)

        return self.loss(pred, target) if not self.inverse else -1 * self.loss(pred, target)