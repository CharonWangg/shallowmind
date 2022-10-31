import torch
from ..builder import OPTIMIZERS


# TODO: many redundancies, need integrate to one BaseOptimizer class


def exclude_norm_and_bias(model):
    '''
    Exclude bias and norm layers from weight decay
    '''
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith('.bias'):
            no_decay.append(param)
        else:
            decay.append(param)

    return no_decay, decay


@OPTIMIZERS.register_module()
class SGD(torch.optim.SGD):
    def __init__(self, model, weight_decay=5e-4, filter_norm_and_bias=False, **kwargs):
        if filter_norm_and_bias:
            no_decay, decay = exclude_norm_and_bias(model)
            params = [{"params": decay,
                       "weight_decay": weight_decay
                       },
                      {"params": no_decay,
                       "weight_decay": 0.0
                       }]
        else:
            params = model.parameters()
        super(SGD, self).__init__(params=params, **kwargs)


@OPTIMIZERS.register_module()
class Adam(torch.optim.Adam):
    def __init__(self, model, weight_decay=5e-4, filter_norm_and_bias=False, **kwargs):
        if filter_norm_and_bias:
            no_decay, decay = exclude_norm_and_bias(model)
            params = [{"params": decay,
                       "weight_decay": weight_decay
                       },
                      {"params": no_decay,
                       "weight_decay": 0.0
                       }]
        else:
            params = model.parameters()
        super(Adam, self).__init__(params=params, **kwargs)


@OPTIMIZERS.register_module()
class AdamW(torch.optim.AdamW):
    def __init__(self, model, weight_decay=5e-4, filter_norm_and_bias=False, **kwargs):
        if filter_norm_and_bias:
            no_decay, decay = exclude_norm_and_bias(model)
            params = [{"params": decay,
                       "weight_decay": weight_decay
                       },
                      {"params": no_decay,
                       "weight_decay": 0.0
                       }]
        else:
            params = model.parameters()
        super(AdamW, self).__init__(params=params, **kwargs)


@OPTIMIZERS.register_module()
class RMSprop(torch.optim.RMSprop):
    def __init__(self, model, weight_decay=5e-4, filter_norm_and_bias=False, **kwargs):
        if filter_norm_and_bias:
            no_decay, decay = exclude_norm_and_bias(model)
            params = [{"params": decay,
                       "weight_decay": weight_decay
                       },
                      {"params": no_decay,
                       "weight_decay": 0.0
                       }]
        else:
            params = model.parameters()
        super(RMSprop, self).__init__(params=params, **kwargs)


@OPTIMIZERS.register_module()
class Bert(torch.optim.AdamW):
    def __init__(self, model, weight_decay=5e-4, eps=1e-08, correct_bias=True,
                 filter_norm_and_bias=True, **kwargs):
        if filter_norm_and_bias:
            no_decay, decay = exclude_norm_and_bias(model)
            params = [{"params": decay,
                       "weight_decay": weight_decay
                       },
                      {"params": no_decay,
                       "weight_decay": 0.0
                       }]
        else:
            params = model.parameters()

        super(Bert, self).__init__(params=params, weight_decay=weight_decay, eps=eps, correct_bias=correct_bias,
                                   **kwargs)


@OPTIMIZERS.register_module()
class TorchOptimizer:
    def __init__(self, model, optimizer_name='SGD', weight_decay=5e-4,
                 filter_norm_and_bias=False, **kwargs):
        if filter_norm_and_bias:
            no_decay, decay = exclude_norm_and_bias(model)
            params = [{"params": decay,
                       "weight_decay": weight_decay
                       },
                      {"params": no_decay,
                       "weight_decay": 0.0
                       }]
        else:
            params = model.parameters()
        self.optimizer = getattr(torch.optim, optimizer_name)(params=params, weight_decay=weight_decay, **kwargs)

    def step(self, closure=None):
        self.optimizer.step(closure=closure)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def __getattr__(self, name):
        return getattr(self.optimizer, name)

    def __repr__(self):
        return self.optimizer.__repr__()

    def __str__(self):
        return self.optimizer.__str__()

    def __dir__(self):
        return self.optimizer.__dir__()


