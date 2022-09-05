import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from ..builder import BACKBONES, build_backbone, build_head
from einops import rearrange


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

class PatchEmbedding(pl.LightningModule):
    PAD_IDX = -100
    # embed causal windows into a fixed-size embedding space
    def __init__(self, window_encoder):
        super().__init__()
        self.window_encoder = nn.Sequential(
            # improve efficiency by using a convolutional layer instead of a linear layer
            build_backbone(window_encoder.backbone),
            build_head(window_encoder.head)
        )
        self.num_windows = window_encoder.num_windows
        self.embedding_size = window_encoder.head.num_classes
        self.position_encoding = PositionalEncoding(window_encoder.head.num_classes,
                                                    window_encoder.num_windows)

    def forward(self, x):
        x_padding_mask = x['padding_mask']
        x_mask = torch.zeros((x_padding_mask.shape[1], x_padding_mask.shape[1])).to(torch.bool).to(self.device)
        x = x['seq']

        # (B, NW, WL, C) -> (B, NW, C)
        b, nw, wl, c = x.shape
        # x = torch.stack([self.window_encoder(xx[torch.where(x_padding_mask[i]!=self.PAD_IDX)[0]]) for i, xx in enumerate(x)], dim=0)
        all_mask = rearrange(x_padding_mask, 'B NW -> (B NW)')
        x = rearrange(x, 'B NW WL C -> (B NW) WL C')
        x = iter(self.window_encoder(x[all_mask==False]))
        x = torch.stack([next(x) if mask==False else self.PAD_IDX * torch.ones(self.embedding_size).to(self.device)
                         for mask in all_mask], dim=0).reshape(b, nw, -1)
        # position encoding
        x = self.position_encoding(x)

        return x, x_mask, x_padding_mask


@BACKBONES.register_module()
class CausalTransformer(pl.LightningModule):
    # use conv1d to embed causal window and feed embedding to transformer
    def __init__(self, window_encoder, hidden_size=None, num_layers=None, nhead=None, dropout=0.1):
        super().__init__()
        self.__dict__.update(locals())
        self.window_encoder = PatchEmbedding(window_encoder)
        embedding_size = window_encoder.head.num_classes
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=nhead,
                                                   dim_feedforward=hidden_size,
                                                   dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x, x_mask, x_padding_mask = self.window_encoder(x)
        x = self.transformer(x, x_mask, x_padding_mask)
        return {'x': [x], 'mask': x_padding_mask}

