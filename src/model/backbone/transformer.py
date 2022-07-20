import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from ..builder import BACKBONES

@BACKBONES.register_module()
class Transformer(pl.LightningModule):
    def __init__(self, input_size=None, hidden_size=None, trm_hidden_size=None,
                 input_length=None, num_layers=None, nhead=None, dropout=0.1,
                 high_dim_projection=True):
        super().__init__()
        self.save_hyperparameters()
        self.__dict__.update(locals())
        self.emb = nn.Linear(input_size, hidden_size) if high_dim_projection else None
        self.pos_encoder = PositionalEncoding(hidden_size, input_length)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead,
                                                   dim_feedforward=trm_hidden_size,
                                                   dropout=dropout, batch_first=True)
        self.trm = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # high level projection:
        if self.emb is not None:
            x = self.emb(x)
        # positional encoding:
        x = self.pos_encoder(x)
        # transformer:
        x = self.trm(x)
        return [x]


class PositionalEncoding(nn.Module):

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

