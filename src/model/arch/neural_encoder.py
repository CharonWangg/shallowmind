import torch
import torch.nn as nn
import pytorch_lightning as pl
from einops import repeat
from ..utils import add_prefix
from ..builder import ARCHS
from ..builder import build_backbone, build_head
from ...data.pipeline import Compose
from .base_encoder_decoder import BaseEncoderDecoder


@ARCHS.register_module()
class NeuralEncoder(BaseEncoderDecoder):
    def __init__(self, backbone, head, auxiliary_head=None, pipeline=None):
        _5D = backbone.pop('_5D', False)
        super(NeuralEncoder, self).__init__(backbone, head, auxiliary_head, pipeline)
        if _5D:
            self.name = 'NeuralEncoder5D'
            self.backbone.model.conv1 = nn.Conv2d(5, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        else:
            self.name = 'NeuralEncoder2D'

    def exact_feat(self, x):
        if x.dim() == 4:
            x = repeat(x, 'b h w s -> b c h w s', c=3)
        # extract the feature for different subject
        x = torch.concat([self.backbone(x[..., i])[-1] for i in range(x.shape[-1])], dim=-1)

        return [x]
