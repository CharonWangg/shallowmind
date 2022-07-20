import torch
import pytorch_lightning as pl
from ..builder import LOSSES

@LOSSES.register_module()
class PoissonLoss(pl.LightningModule):
    def __init__(self, bias=1e-12, per_neuron=False, reduction='mean', scale=True, loss_weight=1.0):
        super().__init__()
        assert reduction in ['mean', 'sum'], 'reduction must be either mean or sum'
        self.bias = bias
        self.per_neuron = per_neuron
        self.reduction = reduction
        self.scale = scale
        self.loss_name = 'PoissonLoss'
        self.loss_weight = loss_weight

    def forward(self, output, target):
        target = target.detach()
        loss = output - target * torch.log(output + self.bias)
        if not self.per_neuron:
            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()
        else:
            loss = loss.view(-1, loss.shape[-1])
            if self.reduction == 'mean':
                loss = loss.mean(dim=0)
            elif self.reduction == 'sum':
                loss = loss.sum(dim=0)
        if self.scale:
            loss = loss * (4500 / output.shape[0]) ** 0.5
        return loss


