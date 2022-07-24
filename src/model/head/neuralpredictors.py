import timm
import torch
import torch.nn as nn
import neuralpredictors.layers.readouts as readouts
import pytorch_lightning as pl
from .base import BaseHead
from ..builder import HEADS

def pairwise_squared_euclidean_f_by_n(x):
    xx = torch.sum(x*x, dim=0)
    xy = torch.einsum('...i,...j->ij', x, x)
    return xx.view(1, -1) + xx.view(-1 , 1) - 2*xy

class MultipleReadoutWrapper(readouts.MultiReadoutSharedParametersBase):
    def __init__(self, model_name, **kwargs):
        self._base_readout = getattr(readouts, model_name)
        super(MultipleReadoutWrapper, self).__init__(**kwargs)

@HEADS.register_module()
class NeuralPredictors(BaseHead):
    '''call the readouts in NeuralPredictors library'''
    def __init__(self, model_name, multiple=False, elu_offset=0.0, in_index=-1,
                 losses=dict(type='TorchLoss', loss_name='CrossEntropyLoss', loss_weight=1.0),
                 **kwargs):
        super(NeuralPredictors, self).__init__(in_index=in_index, losses=losses)
        self.elu_offset = elu_offset
        self.spatial_sim = kwargs.pop('spatial_similarity', None)

        if multiple:
            self.model = MultipleReadoutWrapper(model_name, **kwargs)
        else:
            self.model = getattr(readouts, model_name)(**kwargs)

        # Optional 'spatial_similiarity' argument is a #neurons x #neurons matrix, where sim[i,j] is a weight on
        #   how strongly we regularize that weight[i] == weight[j]
        if self.spatial_sim is not None:
            self.tikhonov_reg = True
            for key in self.model.keys():
                assert key in self.spatial_sim, f"key {key} not in spatial_similarity dict!"
        else:
            self.tikhonov_reg = True


    def forward(self, x, **kwargs):
        '''add elu to prevent negative values'''
        return nn.functional.elu(self.model(x[self.in_index], **kwargs) + self.elu_offset) + 1

    def regularizer(self, data_key=None, reduction="sum", average=None):
        reg = self.model.regularizer(data_key, reduction, average)

        if data_key is None and len(self) == 1:
            data_key = list(self.keys())[0]

        # TODO (?) move to per-readout regularizer?
        if self.spatial_sim is not None:
            sim = self.spatial_sim[data_key].to(self.device)
            readout = self.model[data_key]
            readout_features_f_by_n = readout.features.squeeze()
            diff2 = pairwise_squared_euclidean_f_by_n(readout_features_f_by_n)
            diff2_loss = torch.sum(torch.triu(sim * diff2, 1))
            if average:
                n_neurons = readout_features_f_by_n.shape[-1]
                n_pairs = n_neurons * (n_neurons - 1) / 2
                diff2_loss = diff2_loss / n_pairs
            reg += diff2_loss

        return reg

