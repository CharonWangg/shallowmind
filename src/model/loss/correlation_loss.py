import torch
import torch.nn as nn
import pytorch_lightning as pl
from ..builder import LOSSES

@LOSSES.register_module()
class CorrelationLoss(pl.LightningModule):
    def __init__(self, loss_name='CorrelationLoss', reduction='mean', eps=1e-12, loss_weight=1.0):
        super().__init__()
        assert reduction in ['mean', 'sum'], 'reduction must be either mean or sum.'
        self.loss_name = loss_name
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.cos = nn.CosineSimilarity(dim=-1, eps=eps)

    def forward(self, output, target):
        target = target.detach()
        if self.reduction == 'mean':
            pearson = self.cos(output - output.mean(dim=0, keepdim=True), target - target.mean(dim=0, keepdim=True))
            return (1 - pearson ** 2).mean()
        elif self.reduction == 'sum':
            pearson = self.cos(output - output.mean(dim=0, keepdim=True), target - target.mean(dim=0, keepdim=True))
            return (1 - pearson ** 2).sum()
