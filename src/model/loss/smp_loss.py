import pytorch_lightning as pl
from ..builder import LOSSES
import segmentation_models_pytorch as smp


@LOSSES.register_module()
class SMPLoss(pl.LightningModule):
    def __init__(self, loss_name=None, loss_weight=1.0, **kwargs):
        super(SMPLoss, self).__init__()
        if loss_name is None:
            raise ValueError('loss_name is required')
        self.loss_name = loss_name
        self.loss_weight = loss_weight
        self.loss = getattr(smp, self.loss_name)(**kwargs)

    def forward(self, pred, target):
        return self.loss(pred, target)
