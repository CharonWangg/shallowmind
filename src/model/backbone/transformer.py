import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from ..builder import BACKBONES, build_embedding
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat

@BACKBONES.register_module()
class Transformer(pl.LightningModule):
    def __init__(self, in_channels=None, hidden_size=None,
                 input_length=None, num_layers=None, nhead=None, dropout=0.1,
                 embedding=None):
        super().__init__()
        self.__dict__.update(locals())

        if embedding is not None:
            self.embedding = build_embedding(embedding)
            if in_channels is None:
                in_channels = self.embedding.embedding_size
        else:
            self.embedding = None

        encoder_layer = nn.TransformerEncoderLayer(d_model=in_channels, nhead=nhead,
                                                   dim_feedforward=hidden_size,
                                                   dropout=dropout, batch_first=True)
        self.trm = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


    def forward(self, x):
        if isinstance(x, dict):
            assert 'image' or 'seq' in x.keys(), 'input must be a dict with key "image" or "seq"'
            x = x['image'] if 'image' in x.keys() else x['seq']
        # high level projection:
        if self.embedding is not None:
            x = self.embedding(x)
        # transformer:
        x = self.trm(x)
        return [x]

class PositionalEncoding(nn.Module):
    # sinusoidal positional encoding
    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

