import torch.nn as nn
import torch.nn.functional as F
from neuralpredictors.layers import readouts
from ..builder import HEADS
from .base import BaseHead
from .neuralpredictors import MultipleReadoutWrapper

@HEADS.register_module()
class ImageMappingReadout(BaseHead):
    """
    A readout using a spatial transformer layer whose positions are sampled from one Gaussian per neuron. Mean
    and covariance of that Gaussian are learned. In addition, there is an image dependent remapping of neurons
    locations.

    For most parameters see:  FullGaussian2d

    Args:
        remap_layers (int): number of layers of the remapping network
        remap_kernel (int): conv kernel size of the remapping network
        max_remap_amplitude (int): maximal amplitude of remapping (factor on output of remapping network)
    """

    def __init__(self, multiple=True, remap_layers=2, remap_kernel=3, max_remap_amplitude=0.2, elu_offset=0.0,
                 in_index=-1, losses=dict(type='TorchLoss', loss_name='CrossEntropyLoss', loss_weight=1.0),
                 **kwargs):
        super().__init__(in_index=in_index, losses=losses)
        self.in_index = in_index
        self.elu_offset = elu_offset
        if multiple:
            self.model = MultipleReadoutWrapper('RemappedGaussian2d',
                                                  remap_layers=remap_layers,
                                                  remap_kernel=remap_kernel,
                                                  max_remap_amplitude=max_remap_amplitude,
                                                  **kwargs)
        else:
            self.readout = readouts.RemappedGaussian2d(remap_layers=remap_layers,
                                                       remap_kernel=remap_kernel,
                                                       max_remap_amplitude=max_remap_amplitude,
                                                       **kwargs)

        # TODO: modify readout structure

    def forward(self, x, **kwargs):
        '''add elu to prevent negative values'''
        return nn.functional.elu(self.model(x[self.in_index], **kwargs) + self.elu_offset) + 1

    def regularizer(self, **kwargs):
        return self.model.regularizer(**kwargs)