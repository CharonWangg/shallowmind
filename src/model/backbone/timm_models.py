import timm
import torch.nn as nn
import pytorch_lightning as pl
from ..builder import BACKBONES

@BACKBONES.register_module()
class TimmModels(pl.LightningModule):
    '''call the backbones in TIMM library'''
    def __init__(self, model_name, features_only=True, pretrained=True,
                 checkpoint_path='', in_channels=3, num_classes=1000, **kwargs):
        super(TimmModels, self).__init__()
        # TODO: add support for switching to not default normalization
        self.model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            features_only=features_only,
            in_chans=in_channels,
            num_classes=num_classes,
            checkpoint_path=checkpoint_path,
            **kwargs,
        )

        # Remove layers not belongs to backbone
        if features_only:
            self.feature_channels = self.model.feature_info.channels()
            self.model.pool = None
            self.model.fc = None
            self.model.classifier = None
        else:
            self.features_only = features_only
            self.feature_channels = [info['num_chs'] for info in self.model.feature_info]


    def forward(self, x):
        x = self.model(x)
        return x
