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

@LOSSES.register_module()
class PoissonLikeGaussianLoss(pl.LightningModule):
    def __init__(self, bias=1e-1, per_neuron=False, reduction='mean', scale=True, loss_weight=1.0):
        """
        Computes Poisson-like Gaussian loss (squared error normalized by variance, where variance = mean like in a
        Poisson)
        Implemented by Richard Lange but largely copied from PoissonLoss
        Args:
            bias (float, optional): Value used to numerically stabilize evalution of the log-likelihood. Added to variance (denominator of log loss)
            per_neuron (bool, optional): If set to True, the average/total Poisson loss is returned for each entry of the last dimension (assumed to be enumeration neurons) separately. Defaults to False.
            avg (bool, optional): If set to True, return mean loss. Otherwise returns the sum of loss. Defaults to True.
        """
        super().__init__()
        self.bias = bias
        self.per_neuron = per_neuron
        self.per_neuron = per_neuron
        self.reduction = reduction
        self.scale = scale
        self.loss_name = 'PoissonLikeGaussianLoss'
        self.loss_weight = loss_weight

    def forward(self, output, target):
        target = target.detach()
        variance = torch.clip(output, 0., None) + self.bias
        # loss is negative log probability under a gaussian with mean 'output' and variance 'output+bias', but with
        # output clipped so that variance is at least 'bias'
        loss = 1 / 2 * (output - target) ** 2 / variance + 1 / 2 * torch.log(variance).sum()
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


