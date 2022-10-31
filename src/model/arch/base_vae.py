import torch
import torch.nn as nn
from ..utils import add_prefix
from ..builder import ARCHS, build_backbone, build_head, build_arch
from .base import BaseArch
from einops.layers.torch import Rearrange


@ARCHS.register_module()
class BaseVAE(BaseArch):
    def __init__(self, encoder, decoder, **kwargs):
        super(BaseVAE, self).__init__(**kwargs)
        assert encoder is not None, 'encoder is not defined'
        assert decoder is not None, 'decoder is not defined'
        self.name = 'BaseVAE'
        # build backbone
        self.backbone = build_backbone(encoder.backbone)
        # mu and log_var
        encoder.head.in_channels = self.infer_input_shape_for_head(encoder.head)
        self.mu = build_head(encoder.head)
        self.log_var = build_head(encoder.head)
        # build decoder
        decoder = build_head(decoder)
        self.decoder = nn.ModuleDict(
            {'noise_proj': nn.Sequential(nn.Linear(self.mu.num_classes, 32 * 32),
                                         nn.BatchNorm1d(32 * 32),
                                         nn.ReLU(),
                                         nn.Linear(32 * 32, self.mu.num_classes * 7 * 7),
                                         nn.BatchNorm1d(self.mu.num_classes * 7 * 7),
                                         nn.ReLU(),
                                         Rearrange('b (c h w)-> b c h w', h=7, w=7)
                                         ),
             'generator_nn': decoder
             })

        # pop out dataloader
        self.cleanup()

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def forward_train(self, x, label):
        if isinstance(x, dict):
            assert 'image' or 'seq' in x.keys(), 'input must be a dict with key "image" or "seq"'
            x = x['image'] if 'image' in x.keys() else x['seq']
        real_x = x.clone()
        x = self.exact_feat(x)
        mu, log_var = self.mu(x), self.log_var(x)
        p, q, z = self.reparameterize(mu, log_var)
        x = self.decoder['noise_proj'](z)
        x = self.decoder['generator_nn']([x])

        # regularization (kl divergence)
        reg_loss = self.mu.parse_losses(p, q)
        # reconstruction loss
        rec_loss = self.decoder['generator_nn'].parse_losses(x, real_x)

        loss = {**reg_loss, **rec_loss}
        loss['loss'] = sum([loss[k] for k in loss.keys() if 'loss' in k.lower()])
        # pack the output and losses
        return loss

    def forward_test(self, x, label=None):
        if isinstance(x, dict):
            assert 'image' or 'seq' in x.keys(), 'input must be a dict with key "image" or "seq"'
            x = x['image'] if 'image' in x.keys() else x['seq']
        noise_sample = torch.randn(x.shape[0], self.mu.num_classes).to(self.device)
        x = self.decoder['noise_proj'](noise_sample)
        res = {'output': self.decoder['generator_nn']([x])}
        return res
