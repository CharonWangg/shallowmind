import torch
import pytorch_lightning as pl
from ..builder import METRICS
@METRICS.register_module()
class Correlation(pl.LightningModule):
    def __init__(self, metric_name='correlation', eps=1e-12, detach_target=True):
        """
        Compute correlation between the output and the target

        Args:
            eps (float, optional): Used to offset the computed variance to provide numerical stability.
                Defaults to 1e-12.
            detach_target (bool, optional): If True, `target` tensor is detached prior to computation. Appropriate when
                using this as a loss to train on. Defaults to True.
        """
        super().__init__()
        self.metric_name = metric_name
        self.eps = eps
        self.detach_target = detach_target

    def forward(self, output, target):
        if self.detach_target:
            target = target.detach()
        delta_out = output - output.mean(0, keepdim=True)
        delta_target = target - target.mean(0, keepdim=True)

        var_out = delta_out.pow(2).mean(0, keepdim=True)
        var_target = delta_target.pow(2).mean(0, keepdim=True)

        corrs = (delta_out * delta_target).mean(0, keepdim=True) / (
            (var_out + self.eps) * (var_target + self.eps)
        ).sqrt()
        return corrs.mean()

@METRICS.register_module()
class AverageCorrelation(pl.LightningModule):
    def __init__(self, metric_name='average_correlation', by=None, eps=1e-12, detach_target=True):
        """
        Compute correlation between the output and the target

        Args:
            eps (float, optional): Used to offset the computed variance to provide numerical stability.
                Defaults to 1e-12.
            detach_target (bool, optional): If True, `target` tensor is detached prior to computation. Appropriate when
                using this as a loss to train on. Defaults to True.
        """
        super().__init__()
        if by is None or not isinstance(by, str):
            raise ValueError('by must be a string')
        self.metric_name = metric_name
        self.by = by
        self._metric = Correlation(eps=eps, detach_target=detach_target)

    def forward(self, output, target, meta_data):
        if meta_data is None:
            return self._metric(output, target)
        else:
            needed_meta_data = torch.stack([data[self.by] for data in meta_data]).squeeze()
            groups = []
            for group in torch.unique(needed_meta_data):
                groups.append((output[needed_meta_data == group].mean(dim=0), target[needed_meta_data == group].mean(dim=0)))
            res = torch.stack([self._metric(group[0], group[1]) for group in groups]).mean()
            return res