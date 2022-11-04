import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from ..builder import LOSSES


@LOSSES.register_module()
class CosineSimilarityLoss(pl.LightningModule):
    def __init__(self, loss_name='CosineSimilarityLoss', reduction='sum', dim=1, p=2, loss_weight=1.0):
        super(CosineSimilarityLoss, self).__init__()
        assert reduction in ['mean', 'sum'], 'reduction must be either mean or sum.'
        self.loss_name = loss_name
        self.dim = dim
        self.p = p
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, output, target):
        output = F.normalize(output, dim=self.dim, p=self.p)
        target = F.normalize(target, dim=self.dim, p=self.p)
        return 2 - 2 * (output * target).mean(dim=self.dim) if self.reduction == 'mean' \
            else 2 - 2 * (output * target).sum(dim=self.dim).sum()


@LOSSES.register_module()
class SimSiamLoss(pl.LightningModule):
    def __init__(self, loss_name='SimSiamLoss', loss_weight=1.0, **kwargs):
        super(SimSiamLoss, self).__init__()
        self.loss_name = loss_name
        self.loss_weight = loss_weight
        self.loss = CosineSimilarityLoss(**kwargs)

    def forward(self, p1, p2, z1, z2):
        return (self.loss(p1, z2.detach()) + self.loss(p2, z1.detach())) / 2
