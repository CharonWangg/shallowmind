import torch
import torch.nn as nn
from einops import repeat
from ..builder import ARCHS, build_embedding, build_backbone
from .base_encoder_decoder import BaseEncoderDecoder
from einops.layers.torch import Rearrange, Reduce
from ..utils.commons import ResidualAdd


@ARCHS.register_module()
class CIEncoder(BaseEncoderDecoder):
    def __init__(self, backbone, head, auxiliary_head=None, pipeline=None):
        super(CIEncoder, self).__init__(backbone, head, auxiliary_head, pipeline)
        self.embedding = nn.ModuleDict({
                                        'variable_wise': nn.Sequential(
                                                        # nn.LayerNorm(4),
                                                        Rearrange('b m n -> b n m'),
                                                        nn.BatchNorm1d(4),
                                                        nn.Conv1d(4, 64, kernel_size=5, stride=5),
                                                        Rearrange('b n m -> b m n'),
                                                        )})

        # self.embedding = nn.Sequential(*[
        #                                 # mini-batch embedding
        #                                 Rearrange('b n m -> b m n'),
        #                                 nn.Conv1d(4, 64, kernel_size=100, stride=100),
        #                                 Rearrange('b n m -> b m n'),
        #                                 ResidualAdd(nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=256)),
        #                                 Reduce('b m n -> b n', 'mean'),
        # ])

        # self.backbone = nn.Sequential(*[nn.BatchNorm1d(64),
        #                                 nn.Linear(64, 128),
        #                                 nn.GELU(),
        #                                 nn.BatchNorm1d(128),
        #                                 nn.Linear(128, 256),
        #                                 nn.GELU(),
        #                                 nn.BatchNorm1d(256),
        #                                 nn.Linear(256, 256),
        #                                 nn.GELU(),
        #                                 ])
        backbone.pop('embedding')
        self.backbone = build_backbone(backbone)
        self.backbone.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=1)

    def exact_feat(self, x):
        if isinstance(x, dict):
            x = x['seq']
        # if x.dim() == 3:
        #     # (n_batch, mini_batch_size, n_variables) -> (n_batch, 1, mini_batch_size, n_variables): for ImageEncoder
        #     x = repeat(x, 'b h w  -> b c h w ', c=3)
        x = self.embedding['variable_wise'](x).squeeze()
        x = repeat(x, 'b h w  -> b c h w ', c=1)
        return self.backbone(x)
