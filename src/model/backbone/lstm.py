import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from ..builder import BACKBONES, build_embedding

@BACKBONES.register_module()
class LSTM(pl.LightningModule):
    def __init__(self, in_channels=None, hidden_size=None, num_layers=2, dropout=0.0, bidirectional=False,
                 batch_first=True, embedding=None):
        super().__init__()
        self.__dict__.update(locals())

        if embedding is not None:
            self.embedding = build_embedding(embedding)
            if in_channels is None:
                in_channels = self.embedding.embedding_size
        else:
            self.embedding = None

        self.model = nn.LSTM(in_channels, hidden_size, num_layers,
                                 batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

    def forward(self, x):
        if isinstance(x, dict):
            assert 'image' or 'seq' in x.keys(), 'input must be a dict with key "image" or "seq"'
            x = x['image'] if 'image' in x.keys() else x['seq']
        # high level projection:
        if self.embedding is not None:
            x = self.embedding(x)
        x, (h, c) = self.model(x)
        return [x]