import torch.nn as nn
import pytorch_lightning as pl
from ..builder import BACKBONES, build_embedding, build_backbone
from einops.layers.torch import Rearrange, Reduce
from ..utils.commons import ResidualAdd


@BACKBONES.register_module()
class CIEncoder(pl.LightningModule):
    def __init__(self, in_channels=4, hidden_size=64, patch_size=10, nhead=4):
        super(CIEncoder, self).__init__()

        self.model = nn.ModuleDict({'stem': nn.Sequential(*[
                                            # mini-batch embedding
                                            Rearrange('b m n -> b n m'),
                                            # feature normalization
                                            nn.BatchNorm1d(in_channels),
                                            # patch embedding
                                            nn.Conv1d(in_channels, hidden_size,
                                                      kernel_size=patch_size, stride=patch_size),
                                            Rearrange('b n m -> b m n'),
                                            # patch encoding
                                            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead,
                                                                       dim_feedforward=hidden_size),
                                            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead,
                                                                       dim_feedforward=hidden_size),
                                            Reduce('b m n -> b n', 'mean'),
                                         ]),
                                    # 'body': nn.Sequential(*[
                                    #         nn.BatchNorm1d(hidden_size),
                                    #         nn.Linear(hidden_size, hidden_size*2),
                                    #         nn.GELU(),
                                    #         nn.BatchNorm1d(hidden_size*2),
                                    #         nn.Linear(hidden_size*2, hidden_size*2),
                                    #         nn.GELU(),
                                    #     ])
                                    })

    def forward(self, x):
        if isinstance(x, dict):
            assert 'image' or 'seq' in x.keys(), 'input must be a dict with key "image" or "seq"'
            x = x['image'] if 'image' in x.keys() else x['seq']
        x = self.model['stem'](x).squeeze()
        # x = self.model['body'](x)
        return [x]

