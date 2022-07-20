import math
import torch
import torch.nn as nn
from ..builder import build_loss, HEADS
from .base import BaseHead

@HEADS.register_module()
class BasePooler(BaseHead):
    def __init__(self, pooler_type='mean', **kwargs):
        assert pooler_type in ['cls', 'mean', 'max', 'attention'], 'pooler_name must be in cls, mean, max and attention'
        self.pooler_type = pooler_type
        super(BasePooler, self).__init__(**kwargs)
        self.activation = nn.Tanh()
        self.model = nn.Sequential(*[nn.Linear(self.in_channels, self.num_classes),
                                     self.activation])

        # for attention pooler
        if pooler_type == 'attention':
            self.q = nn.Linear(self.in_channels, self.in_channels, bias=False)
            self.k = nn.Linear(self.in_channels, self.in_channels, bias=False)
            self.v = nn.Linear(self.in_channels, self.in_channels, bias=False)

    def forward(self, x):
        '''use specific backbone layer output to forward'''
        x = x[self.in_index]
        if self.pooler_type == 'cls':
            pooled_output = x[:, 0]
        elif self.pooler_type == 'mean':
            pooled_output = x.mean(dim=1)
        elif self.pooler_type == 'max':
            pooled_output = x.max(dim=1)
        elif self.pooler_type == 'attention':
            q = self.q(x)
            k = self.k(x)
            v = self.v(x)
            qk = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(self.in_channels)
            attention_scores = qk.softmax(dim=-1)
            pooled_output = torch.matmul(attention_scores, v).sum(dim=1)

        return self.model(pooled_output)
