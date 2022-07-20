import torch.nn as nn
import pytorch_lightning as pl
from ..utils import add_prefix
from ..builder import ARCHS
from ..builder import build_backbone, build_head

@ARCHS.register_module()
class BaseEncoderDecoder(pl.LightningModule):
    def __init__(self, backbone, head, auxiliary_head=None):
        super(BaseEncoderDecoder, self).__init__()
        assert backbone is not None, 'backbone is not defined'
        assert head is not None, 'head is not defined'
        # build backbone
        self.backbone = build_backbone(backbone)
        # build decode head
        if head.get('in_channels', None) is None:
            head.in_channels = self.backbone.feature_channels[head.get('in_index', -1)]
        self.head = build_head(head)
        # build auxiliary head
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for aux_head in auxiliary_head:
                    self.auxiliary_head.append(build_head(aux_head))
            else:
                self.auxiliary_head = nn.ModuleList([build_head(auxiliary_head)])
        else:
            self.auxiliary_head = None

    def exact_feat(self, x):
        x = self.backbone(x)
        return x

    def forward_decode_train(self, feat, label):
        loss = dict()
        decode_loss = self.head.forward_train(feat, label)
        loss.update(add_prefix(f'mainhead', decode_loss))
        return loss

    def forward_auxiliary_train(self, feat, label):
        loss = dict()
        if self.auxiliary_head is not None:
            for idx, auxiliary_head in enumerate(self.auxiliary_head):
                loss.update(add_prefix(f'auxhead{idx}', auxiliary_head.forward_train(feat, label)))
        return loss

    def forward_train(self, x, label):
        loss = dict()
        feat = self.exact_feat(x)

        loss.update(self.forward_decode_train(feat, label))
        loss.update(self.forward_auxiliary_train(feat, label))

        # sum up all losses
        loss.update({'loss': sum([loss[k] for k in loss.keys() if 'loss' in k.lower()])})

        # pack the output and losses
        return loss

    def forward_test(self, x, label=None):
        feat = self.exact_feat(x)
        res = self.head.forward_test(feat, label)

        # sum up all losses
        if label is not None:
            res.update({'loss': sum([res[k] for k in res.keys() if 'loss' in k.lower()])})
        else:
            res.update({'loss': 'Not available'})
        return res







