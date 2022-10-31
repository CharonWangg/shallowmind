import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from ..builder import LOSSES
from torch.autograd import Variable


@LOSSES.register_module()
class KLDivergence(pl.LightningModule):
    # compute KL divergence between two distributions (torch.distributions)
    def __init__(self, loss_name='KLDivergence', reduction='mean', loss_weight=1.0):
        super(KLDivergence, self).__init__()
        self.loss_name = loss_name
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, q, p):
        # directional KL divergence: KL(q||p)
        if self.reduction == 'mean':
            return torch.distributions.kl_divergence(q, p).mean()
        else:
            return torch.distributions.kl_divergence(q, p).sum()
