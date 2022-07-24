import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from ..builder import NECKS
from ..utils import build_conv_layer


@NECKS.register_module()
class BaseNeck(pl.LightningModule):
    def __init__(self, fusion='concat', in_channels=None, num_classes=None,
                       adaptive_sample=True, dim_reduction=None, **kwargs):
        super(BaseNeck, self).__init__()
        assert fusion in ['concat', 'sum', 'mean', 'attention'], 'fusion should be one of concat, sum, mean, attention'
        self.fusion = fusion
        self.in_channels = in_channels
        self.out_channels = num_classes
        self.adaptive_sample = adaptive_sample
        self.dim_reduction = dim_reduction
        if self.dim_reduction is not None:
            assert isinstance(self.dim_reduction, int), 'dim_reduction must be None or an integer.'
            self.dim_reduction_layer = nn.ModuleList([build_conv_layer('Conv2d', in_channels[i], out_channels=dim_reduction, kernel_size=1)
                                                      for i in range(len(in_channels))])


    def adaptive_sample_feature(self, x):
        # resize all feature maps to the largest size
        largest_feature_map_size = max([xx.shape[-2:] for xx in x])
        x = [F.interpolate(xx, size=largest_feature_map_size, mode='nearest', align_corners=None) for xx in x]
        return x

    def forward(self, x, dim=1, **kwargs):

        if self.adaptive_sample:
            # resize all feature maps to the same size
            x = self.adaptive_sample_feature(x)
        if self.dim_reduction:
            x = [self.dim_reduction_layer[i](xx) for i, xx in enumerate(x)]
        if self.fusion == 'concat':
            return [torch.cat(x, dim=dim)]
        elif self.fusion == 'sum':
            return [torch.sum(x, dim=dim)]
        elif self.fusion == 'mean':
            return [torch.mean(x, dim=dim)]
        elif self.fusion == 'attention':
            raise NotImplementedError('Attention fusion is not implemented yet.')



