import torch
import pytorch_lightning as pl
from ..builder import LOSSES

@LOSSES.register_module()
class KLLoss(pl.LightningModule):
    def __init__(self, stat=None, loss_weight=1.0):
        super().__init__()
        self.stat = [torch.tensor(v) for v in stat['26872-17-20']]
        self.loss_name = 'KLLoss'
        self.loss_weight = loss_weight

    def forward(self, output, target):
        mu = output.mean(dim=0)
        std = output.std(dim=0)
        dist = ((mu - self.stat[0].cuda()) ** 2 + (std - self.stat[1].cuda()) ** 2).sum()
        # p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        # q = torch.distributions.Normal(mu, std)

        # # 2. get the probabilities from the equation
        # log_qzx = q.log_prob(z)
        # log_pz = p.log_prob(z)
        #
        # # kl
        # kl = (log_qzx - log_pz)
        # kl = kl.sum(-1)
        # return kl
        return dist