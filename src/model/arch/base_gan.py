import torch
import torch.nn as nn
from ..builder import ARCHS
from ..builder import build_arch, build_head
from ..utils import add_prefix
from .base import BaseArch
from einops.layers.torch import Rearrange


@ARCHS.register_module()
class BaseGAN(BaseArch):
    def __init__(self, generator, discriminator, **kwargs):
        super(BaseGAN, self).__init__(**kwargs)
        assert generator is not None, 'generator is not defined'
        assert discriminator is not None, 'discriminator is not defined'
        self.name = 'BaseGAN'
        # build generator
        self.noise_dim = generator.pop('noise_dim', 100)
        # build noise projector
        generator.in_channels = generator.in_channels // 7 // 7
        generator = build_head(generator)
        noise_projector = nn.Sequential(
                                        nn.Linear(self.noise_dim, 32*32),
                                        nn.BatchNorm1d(32*32),
                                        nn.ReLU(),
                                        nn.Linear(32*32, generator.in_channels * 7 * 7),
                                        nn.BatchNorm1d(generator.in_channels * 7 * 7),
                                        nn.ReLU(),
                                        Rearrange('b (c h w) -> b c h w', h=7, w=7))
        self.generator = nn.ModuleDict({'noise_proj': noise_projector,
                                        'generator_nn': generator})
        # build discriminator
        discriminator.dataloader = self.dataloader
        self.discriminator = build_arch(discriminator)

        # pop out dataloader
        self.cleanup()

    def sample_noise(self, batch_size):
        noise_sample = torch.randn((batch_size, self.noise_dim), device=self.device)
        return noise_sample

    def forward_generator(self, x):
        noise_sample = self.sample_noise(x.shape[0])
        noise_sample = [self.generator['noise_proj'](noise_sample)]
        res = self.generator['generator_nn'](noise_sample)
        return res

    def update_generator(self, x, label):
        # generate fake image
        # maximize E[log(D(G(z)))]
        label = {'main_label': torch.ones((x.shape[0], 1), device=self.device)}
        loss = self.discriminator.forward_train(self.forward_generator(x), label)
        loss.update(add_prefix('gen', {k: v for k, v in loss.items() if k != 'loss'}))
        return loss

    def update_discriminator(self, x, label):
        # maximize E[log(D(x))] + E[log(1 - D(G(z)))]
        # real image classification
        # valid label and category label
        label = {'main_label': torch.ones((x.shape[0], 1), device=self.device),
                 'aux_label': label}
        real_loss = self.discriminator.forward_train(x, label)
        # fake image classification
        label = torch.zeros((x.shape[0], 1), device=self.device)
        # prevent gradient update for generator
        fake_loss = self.discriminator.forward_train(self.forward_generator(x).detach(), label)
        # sum up all losses
        loss = dict()
        loss.update(add_prefix('cls_real', real_loss))
        loss.update(add_prefix('cls_fake', fake_loss))
        loss.update({'loss': (loss['cls_real_loss'] + loss['cls_fake_loss'])/2})
        return loss

    def forward_train(self, x, label, optimizer_idx):
        if isinstance(x, dict):
            assert 'image' or 'seq' in x.keys(), 'input must be a dict with key "image" or "seq"'
            x = x['image'] if 'image' in x.keys() else x['seq']
        if optimizer_idx == 0:
            # update generator
            loss = self.update_generator(x, label)
        elif optimizer_idx == 1:
            # update discriminator
            loss = self.update_discriminator(x, label)
        else:
            raise ValueError('optimizer_idx must be 0 or 1')
        # pack the output and losses
        return loss

    def forward_test(self, x, label=None):
        if isinstance(x, dict):
            assert 'image' or 'seq' in x.keys(), 'input must be a dict with key "image" or "seq"'
            x = x['image'] if 'image' in x.keys() else x['seq']
        noise_sample = self.sample_noise(x.shape[0])
        noise_sample = [self.generator['noise_proj'](noise_sample)]
        res = {'output': self.generator['generator_nn'](noise_sample)}

        # # sum up all losses (no loss for generation)
        # if label is not None:
        #     res.update({'loss': sum([res[k] for k in res.keys() if 'loss' in k.lower()])})
        # else:
        #     res.update({'loss': 'Not available'})
        return res
