import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from ..builder import LOSSES

@LOSSES.register_module()
class FocalLoss(pl.LightningModule):
    def __init__(self, loss_name='FocalLoss', alpha=1, gamma=2, step_reduction=False, loss_weight=1.0):
        super(FocalLoss, self).__init__()
        self.loss_name = loss_name
        self.alpha = alpha
        self.gamma = gamma
        self.step_reduction = step_reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        if pred.dim() > 2:
            if self.step_reduction:
                pred = pred.mean(dim=1)
            else:
                pred = pred.view(-1, pred.size(-1))
                target = target.view(-1)

        bce_loss = F.cross_entropy(pred.squeeze(),  target)
        loss = self.alpha * (1 - torch.exp(-bce_loss)) ** self.gamma * bce_loss
        return loss