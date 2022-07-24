import timm
import torch.nn as nn
import neuralpredictors.layers.readouts as readouts
import pytorch_lightning as pl
from .base import BaseHead
from ..builder import HEADS

class MultipleReadoutWrapper(readouts.MultiReadoutSharedParametersBase):
    def __init__(self, model_name, **kwargs):
        self._base_readout = getattr(readouts, model_name)
        super(MultipleReadoutWrapper, self).__init__(**kwargs)

@HEADS.register_module()
class NeuralPredictors(BaseHead):
    '''call the backbones in NeuralPredictors library'''
    def __init__(self, model_name, multiple=False, elu_offset=0.0, in_channels=32, channels=None,
                 num_classes=2, dropout=0.1, act_cfg=dict(type='ReLU'), norm_cfg=dict(type='BatchNorm1d'), in_index=-1,
                 losses=dict(type='TorchLoss', loss_name='CrossEntropyLoss', loss_weight=1.0),
                 **kwargs):
        super(NeuralPredictors, self).__init__(in_channels=in_channels, channels=channels, num_classes=num_classes,
                                               dropout=dropout, in_index=in_index, act_cfg=act_cfg, norm_cfg=norm_cfg,
                                               losses=losses)
        self.elu_offset = elu_offset
        if multiple:
            self.model = MultipleReadoutWrapper(model_name, **kwargs)
        else:
            self.model = getattr(readouts, model_name)(**kwargs)

    def forward(self, x, **kwargs):
        '''add elu to prevent negative values'''
        return nn.functional.elu(self.model(x[self.in_index], **kwargs) + self.elu_offset) + 1
