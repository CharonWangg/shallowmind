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
        if isinstance(x, dict):
            assert 'image' or 'seq' in x.keys(), 'input must be a dict with key "image" or "seq"'
            x = x['image'] if 'image' in x.keys() else x['seq']
        x = self.model(x)
        if isinstance(x, list) or isinstance(x, tuple):
            return x
        return [x]

    def regularizer(self, **kwargs):
        return self.model.regularizer(**kwargs)
