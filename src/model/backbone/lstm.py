import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from ..builder import BACKBONES

@BACKBONES.register_module()
class LSTM(pl.LightningModule):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers=2, dropout=0.0, bidirectional=False,
                 batch_first=True, high_dim_projection=True):
        super().__init__()
        self.__dict__.update(locals())
        if high_dim_projection:
            self.emb = nn.Linear(input_size, embedding_size) if high_dim_projection else None
            self.model = nn.LSTM(embedding_size, hidden_size, num_layers,
                                batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        else:
            self.emb = None
            self.model = nn.LSTM(input_size, hidden_size, num_layers,
                                 batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

    def forward(self, x):
        # high level projection:
        if self.emb is not None:
            x = self.emb(x)
        x, (h, c) = self.model(x)
        return [x]