import torchmetrics
import pytorch_lightning as pl
from ..builder import METRICS

@METRICS.register_module()
class TorchMetrics(pl.LightningModule):
    def __init__(self, metric_name=None, step_reduction=None, prob=True, **kwargs):
        super(TorchMetrics, self).__init__()
        if metric_name is None:
            raise ValueError('loss_name is required')
        assert step_reduction in ['mean', 'sum', None]
        self.step_reduction = step_reduction
        self.prob = prob
        self.metrics = getattr(torchmetrics, metric_name)(**kwargs)
        self.metric_name = metric_name.lower()

    def forward(self, pred, target):
        if self.prob:
            pred = pred.softmax(dim=-1)
        if self.step_reduction == 'mean':
            pred = pred.mean(dim=1)
        elif self.step_reduction == 'sum':
            pred = pred.sum(dim=1)
        # for binary classification (AUROC)
        if getattr(self.metrics, 'pos_label', None) is not None:
            pred = pred[..., self.metrics.pos_label]
        return self.metrics(pred.detach().cpu(), target.detach().cpu())