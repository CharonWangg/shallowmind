import timm
import torch.nn as nn
import pytorch_lightning as pl
from ..builder import BACKBONES
from ..builder import build_embedding
from ast import literal_eval


@BACKBONES.register_module()
class TimmModels(pl.LightningModule):
    '''call the backbones in TIMM library'''
    def __init__(self, model_name, embedding=None, features_only=True, remove_fc=True, pretrained=True,
                 checkpoint_path='', in_channels=3, num_classes=1000, **kwargs):
        super(TimmModels, self).__init__()
        if embedding is not None:
            self.embedding = build_embedding(embedding)
            if in_channels is None:
                in_channels = self.embedding.embedding_size
        else:
            self.embedding = None

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
        self.features_only = features_only
        if self.features_only:
            self.feature_channels = self.model.feature_info.channels()
            self.model.global_pool = None
            self.model.fc = None
            self.model.classifier = None
        elif not features_only and remove_fc:
            self.model.fc = nn.Identity()
            self.model.classifier = nn.Identity()
            self.feature_channels = [info['num_chs'] for info in self.model.feature_info]
        elif not features_only and not remove_fc:
            self.feature_channels = [info['num_chs'] for info in self.model.feature_info]

    def forward(self, x):
        if isinstance(x, dict):
            assert 'image' or 'seq' in x.keys(), 'input must be a dict with key "image" or "seq"'
            x = x['image'] if 'image' in x.keys() else x['seq']
        if self.embedding is not None:
            x = self.embedding(x)
        x = self.model(x)
        return x if self.features_only else [x]