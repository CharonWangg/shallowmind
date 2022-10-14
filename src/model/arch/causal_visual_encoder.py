import torch.nn as nn
import pytorch_lightning as pl
from ..utils import add_prefix
from ..builder import ARCHS
from ..builder import build_backbone, build_head, build_embedding
from ...data.pipeline import Compose
from einops import repeat, rearrange, reduce
from einops.layers.torch import Rearrange


@ARCHS.register_module()
class CausalVisualEncoder(pl.LightningModule):
    def __init__(self, front_backbone, back_backbone, patch_embedding, head, auxiliary_head, pipeline=None):
        '''
        Image -> front_backbone -> 16*16 patches -> Transformer -> non-causal head -> HSIC
                                                        -> causal head -> similarity
        Image(Perturbed) -> back_backbone -> 16*16 patches -> Transformer -> causal head -> similarity
                                                                  -> non-causal head -> HSIC
        Architecture: backbone -> Rearrange -> Transformer -> head
        '''
        super(CausalVisualEncoder, self).__init__()
        assert front_backbone or back_backbone is not None, 'backbone is not defined'
        assert head is not None, 'head is not defined'
        assert auxiliary_head is not None, 'auxiliary_head is not defined'
        self.name = 'CausalVisualEncoder'
        # build backbone
        self.front_backbone = build_backbone(front_backbone)
        self.back_backbone = build_backbone(back_backbone)
        # build patch embedding
        self.rearrange = nn.Sequential(Rearrange('b c h w -> b (c h w)'),
                                       Rearrange('b (p1 p2 c) -> b (p1 p2) c ',
                                           p1=patch_embedding.get('patch_size', 16),
                                           p2=patch_embedding.get('patch_size', 16))
                                        )
        self.patch_embedding = build_embedding(patch_embedding)
        # build causal head
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
        if x.get('orig_image', None) is not None:
            # get the image and perturbed image
            image, perturbed_image = x['orig_image'], x['aug_image']
            image, perturbed_image = self.front_backbone(image)[-1], self.front_backbone(perturbed_image)[-1]
            # conv the image and perturbed image into 16*16 patches
            image, perturbed_image = self.rearrange(image), self.rearrange(perturbed_image)
            print(image.shape, perturbed_image.shape)
            image, perturbed_image = self.patch_embedding(image), self.patch_embedding(perturbed_image)
            return image, perturbed_image
        else:
            # get only orig image for test
            image = x['image']
            image = self.front_backbone(image)
            # conv the image and perturbed image into 16*16 patches
            return self.patch_embedding(image)

    def causal_discovery(self, orig_feat, perturb_feat):
        # infer the adjacency matrix of patches
        orig_adjacency_matrix = self.back_backbone(orig_feat)[-1]
        perturbed_adjacency_matrix = self.back_backbone(perturb_feat)[-1]
        orig_adjacency_matrix = reduce(orig_adjacency_matrix, 'b l c -> b l', reduction='mean')
        perturbed_adjacency_matrix = reduce(perturbed_adjacency_matrix, 'b l c -> b l', reduction='mean')
        return orig_adjacency_matrix, perturbed_adjacency_matrix

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
        orig_feat, perturbed_feat = self.exact_feat(x)
        orig_adjacency_matrix, perturbed_adjacency_matrix = self.causal_discovery(orig_feat, perturbed_feat)
        print(orig_adjacency_matrix.shape, perturbed_adjacency_matrix.shape)

        # minimize the MSE between the causal feature and maximize the HSIC between it and non-causal feature
        loss.update(self.forward_decode_train(orig_adjacency_matrix * orig_feat,
                                              perturbed_adjacency_matrix * perturbed_feat))
        loss.update(self.forward_auxiliary_train(~orig_adjacency_matrix * orig_feat,
                                                 orig_adjacency_matrix * orig_feat))
        loss.update(self.forward_auxiliary_train(perturbed_adjacency_matrix * perturbed_feat,
                                                 ~perturbed_adjacency_matrix * perturbed_feat))

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
