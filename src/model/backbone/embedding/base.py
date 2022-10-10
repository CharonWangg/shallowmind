import torch
import torch.nn as nn
import pytorch_lightning as pl
from ...builder import EMBEDDINGS
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat


class Norm(pl.LightningModule):
    '''base class for all basic norm layers in the pytorch'''

    def __init__(self, norm_cfg=dict(type='BatchNorm1d', reshape_list=None), **kwargs):
        super(Norm, self).__init__()
        norm_type = norm_cfg.pop('type', None)
        # reshape (None, list(original_shape, reshaped_shape)) -> reshape the tensor for doing correct normalization
        # ['b l c', 'b c l']
        self.reshape_list = norm_cfg.pop('reshape_list', None)
        if self.reshape_list is not None:
            assert isinstance(self.reshape_list,
                              list), 'reshape_list must be None or a list of two strings indicate the orig and reshaped shape'
            self.reshape = Rearrange(f'{self.reshape_list[0]} -> {self.reshape_list[1]}')
            self.reshape_back = Rearrange(f'{self.reshape_list[1]} -> {self.reshape_list[0]}')
        if norm_type is not None:
            self.model = getattr(nn, norm_type)(**norm_cfg)
        else:
            self.model = nn.Identity()

    def forward(self, x):
        if self.reshape_list is not None:
            x = self.reshape(x)
            x = self.model(x)
            x = self.reshape_back(x)
        else:
            x = self.model(x)
        return x


@EMBEDDINGS.register_module()
class BaseEmbedding(pl.LightningModule):
    '''base class for all basic embedding layers in the pytorch'''

    def __init__(self, input_length, embedding_size, position=False, input_norm=None, reshape_list=None, **kwargs):
        super(BaseEmbedding, self).__init__()
        self.input_length = input_length
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(input_length, embedding_size, **kwargs)
        self.position = position
        self.reshape_list = reshape_list
        if self.reshape_list is not None:
            assert isinstance(self.reshape_list,
                              list), 'reshape_list must be None or a list of two strings indicate the orig and reshaped shape'
            self.reshape = Rearrange(f'{self.reshape_list[0]} -> {self.reshape_list[1]}')
            self.reshape_back = Rearrange(f'{self.reshape_list[1]} -> {self.reshape_list[0]}')
        if position:
            self.positions = nn.Parameter(torch.randn(input_length, embedding_size))
        if input_norm is not None:
            self.input_norm = Norm(input_norm)
        else:
            self.input_norm = None

    def forward(self, x):
        if self.input_norm is not None:
            x = self.input_norm(x)
        if self.reshape_list is not None:
            x = self.reshape(x)
            x = self.embedding(x)
            x = self.reshape_back(x)
        else:
            x = self.embedding(x)
        if self.position:
            x += self.positions
        return x


@EMBEDDINGS.register_module()
class LinearEmbedding(pl.LightningModule):
    '''linear embedding layer'''

    def __init__(self, input_length, embedding_size, position=False, input_norm=None, reshape_list=None, **kwargs):
        super(LinearEmbedding, self).__init__()
        self.input_length = input_length
        self.embedding_size = embedding_size
        self.embedding = nn.Linear(input_length, embedding_size, **kwargs)
        self.position = position
        self.reshape_list = reshape_list
        if self.reshape_list is not None:
            assert isinstance(self.reshape_list,
                              list), 'reshape_list must be None or a list of two strings indicate the orig and reshaped shape'
            self.reshape = Rearrange(f'{self.reshape_list[0]} -> {self.reshape_list[1]}')
            self.reshape_back = Rearrange(f'{self.reshape_list[1]} -> {self.reshape_list[0]}')
        if position:
            self.positions = nn.Parameter(torch.randn(input_length, embedding_size))
        if input_norm is not None:
            self.input_norm = Norm(input_norm)
        else:
            self.input_norm = None

    def forward(self, x):
        if self.input_norm is not None:
            x = self.input_norm(x)
        if self.reshape_list is not None:
            x = self.reshape(x)
            x = self.embedding(x)
            x = self.reshape_back(x)
        else:
            x = self.embedding(x)
        if self.position:
            x += self.positions
        return x


@EMBEDDINGS.register_module()
class ConvEmbedding(pl.LightningModule):
    '''conv embedding layer'''
    def __init__(self, input_length, embedding_size, position=False, input_norm=None, mode='1d',
                 reshape_list=None, **kwargs):
        super(ConvEmbedding, self).__init__()
        self.input_length = input_length
        self.embedding_size = embedding_size
        self.reshape_list = reshape_list
        if mode == '1d':
            self.embedding = nn.Conv1d(input_length, embedding_size, **kwargs)
        elif mode == '2d':
            self.embedding = nn.Conv2d(input_length, embedding_size, **kwargs)
        elif mode == '3d':
            self.embedding = nn.Conv3d(input_length, embedding_size, **kwargs)
        else:
            raise NotImplementedError
        if self.reshape_list is not None:
            assert isinstance(self.reshape_list,
                              list), 'reshape_list must be None or a list of two strings indicate the orig and reshaped shape'
            self.reshape = Rearrange(f'{self.reshape_list[0]} -> {self.reshape_list[1]}')
            self.reshape_back = Rearrange(f'{self.reshape_list[1]} -> {self.reshape_list[0]}')
        self.position = position
        if position:
            self.positions = nn.Parameter(torch.randn(input_length, embedding_size))
        if input_norm is not None:
            self.input_norm = Norm(input_norm)
        else:
            self.input_norm = None

    def forward(self, x):
        if self.input_norm is not None:
            x = self.input_norm(x)
        if self.reshape_list is not None:
            x = self.reshape(x)
            x = self.embedding(x)
            x = self.reshape_back(x)
        else:
            x = self.embedding(x)
        if self.position:
            x += self.positions
        return x


@EMBEDDINGS.register_module()
class PositionEmbedding(pl.LightningModule):
    '''trainable positional embedding layer'''

    def __init__(self, input_length, embedding_size, reshape_list=None, **kwargs):
        super(PositionEmbedding, self).__init__()
        self.positions = nn.Parameter(torch.randn(input_length, embedding_size))
        self.input_length = input_length
        self.reshape_list = reshape_list
        if self.reshape_list is not None:
            assert isinstance(self.reshape_list,
                              list), 'reshape_list must be None or a list of two strings indicate the orig and reshaped shape'
            self.reshape = Rearrange(f'{self.reshape_list[0]} -> {self.reshape_list[1]}')
            self.reshape_back = Rearrange(f'{self.reshape_list[1]} -> {self.reshape_list[0]}')

    def forward(self, x):
        if self.reshape_list is not None:
            x = self.reshape(x)
            x += self.positions
            x = self.reshape_back(x)
        else:
            x += self.positions
        return x


@EMBEDDINGS.register_module()
class PatchEmbedding(pl.LightningModule):
    # PatchEmbedding is a module that takes a sequence and returns an embedding of its patches.
    def __init__(self, in_channels=1, input_length=512, embedding_size=128, patch_size=512, stride=None,
                 input_norm=None, mode='1d', reshape_list=['b l c', 'b c l'], **kwargs):
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
                Rearrange(f'{reshape_list[0]} -> {reshape_list[1]}'),
                nn.Conv1d(in_channels, embedding_size, kernel_size=patch_size, stride=stride, **kwargs),
                Rearrange(f'{reshape_list[1]} -> {reshape_list[0]}'),
            )
            self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_size))
            if stride == patch_size:
                self.positions = nn.Parameter(torch.randn((input_length // patch_size) + 1, embedding_size))
            else:
                self.positions = nn.Parameter(
                    torch.randn(((input_length - patch_size) // stride + 1) + 1, embedding_size))
        elif mode == '2d':
            # 2d patch embedding
            if reshape_list == ['b l c', 'b c l']:
                # 2d patch embedding special case
                reshape_list = ['b (h w) c', 'b c (h w)']

            self.projection = nn.Sequential(
                nn.Conv2d(in_channels, embedding_size, kernel_size=patch_size, stride=stride, **kwargs),
                Rearrange(f'{reshape_list[0]} -> {reshape_list[1]}'),
            )
            self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_size))
            if stride == patch_size:
                self.positions = nn.Parameter(torch.randn((input_length // patch_size) + 1, embedding_size))
            else:
                self.positions = nn.Parameter(
                    torch.randn(((input_length - patch_size) // stride + 1) + 1, embedding_size))

        if input_norm is not None:
            self.input_norm = Norm(input_norm)
        else:
            self.input_norm = None

    def forward(self, x):
        if self.input_norm is not None:
            x = self.input_norm(x)
        b = x.shape[0]
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # add the cls token to the beginning of the embedding
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.positions
        return x