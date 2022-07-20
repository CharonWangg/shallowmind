import timm
import neuralpredictors.layers.cores as cores
import pytorch_lightning as pl
from ..builder import BACKBONES

@BACKBONES.register_module()
class NeuralPredictors(pl.LightningModule):
    '''call the backbones in NeuralPredictors library'''
    def __init__(self, model_name, **kwargs):
        super(NeuralPredictors, self).__init__()
        self.model = getattr(cores, model_name)(**kwargs)

    def forward(self, x):
        x = self.model(x)
        return [x]
