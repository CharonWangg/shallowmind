import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from ..builder import LOSSES
from torch.autograd import Variable


@LOSSES.register_module()
class FocalLoss(pl.LightningModule):
    def __init__(self, loss_name='FocalLoss', alpha=None, gamma=2, reduction='mean', loss_weight=1.0):
        super(FocalLoss, self).__init__()
        self.loss_name = loss_name
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        if pred.dim() > 2:
            pred = pred.view(pred.size(0), pred.size(1), -1)  # N,C,H,W => N,C,H*W
            pred = pred.transpose(1, 2)    # N,C,H*W => N,H*W,C
            pred = pred.contiguous().view(-1, pred.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(pred)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != pred.data.type():
                self.alpha = self.alpha.type_as(pred.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()