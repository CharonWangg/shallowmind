import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from ..builder import LOSSES
from torch.autograd import Variable


@LOSSES.register_module()
class HSICLoss(pl.LightningModule):
    def __init__(self, loss_name='HSICLoss', alpha=None, gamma=2, reduction='mean', loss_weight=1.0):
        super(HSICLoss, self).__init__()
        self.loss_name = loss_name
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        m = pred.shape[0]
        xy = torch.matmul(pred, target)
        h = torch.trace(xy) / m ** 2 + torch.mean(pred) * torch.mean(target) - \
            2 * torch.mean(xy) / m
        return - (h * (m / (-1)) ** 2)