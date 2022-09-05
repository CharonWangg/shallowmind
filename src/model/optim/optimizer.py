import torch
from ..builder import OPTIMIZERS
# try:
#     from ranger21 import Ranger21
# except ImportError:
#     Ranger21 = None

@OPTIMIZERS.register_module()
class SGD(torch.optim.SGD):
    def __init__(self, model, **kwargs):
        params = model.parameters()
        super(SGD, self).__init__(params=params, **kwargs)

@OPTIMIZERS.register_module()
class Adam(torch.optim.Adam):
    def __init__(self, model, **kwargs):
        params = model.parameters()
        super(Adam, self).__init__(params=params, **kwargs)

@OPTIMIZERS.register_module()
class AdamW(torch.optim.AdamW):
    def __init__(self, model, **kwargs):
        params = model.parameters()
        super(AdamW, self).__init__(params=params, **kwargs)

@OPTIMIZERS.register_module()
class RMSprop(torch.optim.RMSprop):
    def __init__(self, model, **kwargs):
        params = model.parameters()
        super(RMSprop, self).__init__(params=params, **kwargs)

# @OPTIMIZERS.register_module()
# class Ranger21(Ranger21):
#     def __init__(self, model, **kwargs):
#         params = model.parameters()
#         num_epochs = kwargs.pop('max_epochs', -1)
#         num_batches_per_epoch = kwargs.pop('max_steps', -1) // num_epochs
#         super(Ranger21, self).__init__(params=params, num_epochs=num_epochs,
#                                      num_batches_per_epoch=num_batches_per_epoch,
#                                      **kwargs)

@OPTIMIZERS.register_module()
class Bert(torch.optim.AdamW):
    def __init__(self, model, weight_decay=5e-4, eps=1e-08, correct_bias=True, no_decay=["bias", "LayerNorm.weight"]):
        params = [
            {
                "params": [p for n, p in model.name_weights if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in model.name_weights if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        super(Bert, self).__init__(params=params, weight_decay=weight_decay, eps=eps, correct_bias=correct_bias)
