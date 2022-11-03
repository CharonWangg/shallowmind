from copy import deepcopy
import torchvision
import pytorch_lightning as pl
from ..builder import METRICS
from ..utils import pascal_case_to_snake_case


@METRICS.register_module()
class BaseMetric(pl.LightningModule):
    def __init__(self, metric_name='BaseMetric', **kwargs):
        super(BaseMetric, self).__init__(**kwargs)
        self.metric_name = pascal_case_to_snake_case(metric_name)

    def forward(self, pred, target=None):
        return pred.mean()
