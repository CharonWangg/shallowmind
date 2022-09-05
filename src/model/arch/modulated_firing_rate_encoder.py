import copy
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from ..utils import add_prefix
from ..builder import ARCHS
from ..builder import build_arch, build_backbone, build_neck, build_head, build_loss
from neuralpredictors.training.context_managers import eval_state


@ARCHS.register_module()
class ModulatedFiringRateEncoder(pl.LightningModule):
    def __init__(self, archs, losses, **kwargs):
        super().__init__()
        self.name = 'ModulatedFiringRateEncoder'
        self.archs = nn.ModuleList()
        for arch in archs:
            self.archs.append(build_arch(arch))
        self.losses = nn.ModuleList()
        if isinstance(losses, dict):
            self.losses.append(build_loss(losses))
        elif isinstance(losses, list):
            for loss in losses:
                self.losses.append(build_loss(loss))
        else:
            raise TypeError(f'losses must be a dict or sequence of dict,\
                      but got {type(losses)}')

    def forward_train(self, x, label):
        _output = torch.ones(1).to(self.device)
        for arch in self.archs:
            arch_name = arch.name
            out = arch.forward_test(x, label)['output'] if 'neuralpredictors' in arch_name.lower() else \
                nn.functional.elu(arch.forward_test(x, label)['output']) + 1
            _output = _output * out

        # modulated output (f(x) * g(x))
        _losses = self.parse_losses(_output, label)
        # pack the output and losses
        _losses.update({'loss': sum([_losses[k] for k in _losses.keys() if 'loss' in k.lower()])})
        return _losses

    def forward_test(self, x, label=None):
        _output = {}
        prod = torch.ones(1).to(self.device)
        for arch in self.archs:
            arch_name = arch.name
            out = arch.forward_test(x, label)['output'] if 'neuralpredictors' in arch_name.lower() else \
                    nn.functional.elu(arch.forward_test(x, label)['output']) + 1
            _output.update({arch_name: out})
            prod = prod * _output[arch_name]

        # modulated output (f(x) * g(x))
        res = dict()

        res.update({'output': prod})
        # sum up all losses
        if label is not None:
            res.update(self.parse_losses(prod, label))
            res.update({'loss': sum([res[k] for k in res.keys() if 'loss' in k.lower()])})
        else:
            res.update({'loss': 'Not available'})
        return res

    def parse_losses(self, pred, label):
        """Compute loss"""
        loss = dict()
        # TODO: add sample weight to loss calculation
        for _loss in self.losses:
            if _loss.loss_name not in loss:
                loss[_loss.loss_name] = _loss(pred, label) * _loss.loss_weight
            else:
                loss[_loss.loss_name] += _loss(pred, label) * _loss.loss_weight

        return loss

