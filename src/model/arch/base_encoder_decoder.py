import torch.nn as nn
from ..builder import ARCHS
from ..builder import build_backbone, build_head
from .base import BaseArch


@ARCHS.register_module()
class BaseEncoderDecoder(BaseArch):
    def __init__(self, backbone, head, auxiliary_head=None, **kwargs):
        super(BaseEncoderDecoder, self).__init__(**kwargs)
        assert backbone is not None, 'backbone is not defined'
        assert head is not None, 'head is not defined'
        self.name = 'BaseEncoderDecoder'
        # build backbone
        self.backbone = build_backbone(backbone)
        # build decode head
        head.in_channels = self.infer_input_shape_for_head(head)
        self.head = build_head(head)
        # build auxiliary head
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for aux_head in auxiliary_head:
                    aux_head.in_channels = self.infer_input_shape_for_head(aux_head)
                    self.auxiliary_head.append(build_head(aux_head))
            else:
                auxiliary_head.in_channels = self.infer_input_shape_for_head(auxiliary_head)
                self.auxiliary_head = nn.ModuleList([build_head(auxiliary_head)])
        else:
            self.auxiliary_head = None

        # pop out dataloader
        self.cleanup()
