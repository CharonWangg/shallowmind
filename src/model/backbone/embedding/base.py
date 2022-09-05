import torch
import torch.nn as nn
import pytorch_lightning as pl
from ...builder import EMBEDDINGS
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat

@EMBEDDINGS.register_module()
class BaseEmbedding(pl.LightningModule):
    '''base class for all basic embedding layers in the pytorch'''
    def __init__(self, input_length, embedding_size, position=False, **kwargs):
        super(BaseEmbedding, self).__init__()
        self.input_length = input_length
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(input_length, embedding_size, **kwargs)
        self.position = position
        if position:
            self.positions = nn.Parameter(torch.randn(input_length, embedding_size))

    def forward(self, x):
        x = self.embedding(x)
        if self.position:
            x += self.positions
        return x

@EMBEDDINGS.register_module()
class LinearEmbedding(pl.LightningModule):
    '''linear embedding layer'''
    def __init__(self, input_length, embedding_size, position=False, **kwargs):
        super(LinearEmbedding, self).__init__()
        self.input_length = input_length
        self.embedding_size = embedding_size
        self.embedding = nn.Linear(input_length, embedding_size, **kwargs)
        self.position = position
        if position:
            self.positions = nn.Parameter(torch.randn(input_length, embedding_size))

    def forward(self, x):
        x = self.embedding(x)
        if self.position:
            x += self.positions
        return x

@EMBEDDINGS.register_module()
class PositionEmbedding(pl.LightningModule):
    '''trainable positional embedding layer'''
    def __init__(self, input_length, embedding_size):
        super(PositionEmbedding, self).__init__()
        self.positions = nn.Parameter(torch.randn(input_length, embedding_size))

    def forward(self, x):
        x += self.positions
        return x

@EMBEDDINGS.register_module()
class PatchEmbedding(pl.LightningModule):
    # PatchEmbedding is a module that takes a sequence and returns an embedding of its patches.
    def __init__(self, in_channels=1, input_length=512, embedding_size=128, patch_size=512, stride=None, mode='1d', **kwargs):
        super().__init__()
        assert mode in ['1d', '2d'], 'mode must be either 1d or 2d'
        self.input_length = input_length
        self.embedding_size = embedding_size
        if stride is None:
            stride = patch_size
        else:
            assert stride > 0, 'stride must be positive'

        if mode == '1d':
            # 1d patch embedding
            self.projection = nn.Sequential(
                Rearrange('b l c -> b c l'),
                nn.Conv1d(in_channels, embedding_size, kernel_size=patch_size, stride=stride, **kwargs),
                Rearrange('b c l -> b l c'),
            )
            self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_size))
            if stride == patch_size:
                self.positions = nn.Parameter(torch.randn((input_length // patch_size) + 1, embedding_size))
            else:
                self.positions = nn.Parameter(torch.randn(((input_length - patch_size) // stride + 1) + 1, embedding_size))
        elif mode == '2d':
            # 2d patch embedding
            self.projection = nn.Sequential(
                nn.Conv2d(in_channels, embedding_size, kernel_size=patch_size, stride=stride, **kwargs),
                Rearrange('b c (h) (w) -> b (h w) c'),
            )
            self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_size))
            if stride == patch_size:
                self.positions = nn.Parameter(torch.randn((input_length // patch_size) + 1, embedding_size))
            else:
                self.positions = nn.Parameter(torch.randn(((input_length - patch_size) // stride + 1) + 1, embedding_size))

    def forward(self, x):
        b = x.shape[0]
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # add the cls token to the beginning of the embedding
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.positions
        return x