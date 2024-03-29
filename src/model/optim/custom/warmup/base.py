# modified from https://github.com/Tony-Y/pytorch_warmup

import math
from contextlib import contextmanager
from torch.optim import Optimizer
from ....builder import SCHEDULERS


def get_warmup_params(period, group_count):
    if type(period) == list:
        if len(period) != group_count:
            raise ValueError(
                'size of period does not equal {}.'.format(group_count))
        for x in period:
            if type(x) != int:
                raise ValueError(
                    'An element in period, {}, is not an int.'.format(
                        type(x).__name__))
        warmup_params = [dict(period=x) for x in period]
    elif type(period) == int:
        warmup_params = [dict(period=period)
                         for _ in range(group_count)]
    else:
        raise TypeError('{} is not a list nor an int.'.format(
            type(period).__name__))
    return warmup_params


@SCHEDULERS.register_module()
class BaseWarmup(object):
    """Base class for all warmup schedules"""
    def __init__(self, optimizer, warmup_params, last_step=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f'{type(optimizer).__name__} is not an Optimizer')
        self.optimizer = optimizer
        self.warmup_params = warmup_params
        self.last_step = last_step
        self.lrs = [group['lr'] for group in self.optimizer.param_groups]
        self.dampen()

    def state_dict(self):
        """Returns the state of the warmup scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the warmup scheduler's state.

        Arguments:
            state_dict (dict): warmup scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def dampen(self, step=None):
        """Dampen the learning rates.

        Arguments:
            step (int): The index of current step. (Default: None)
        """
        if step is None:
            step = self.last_step + 1
        self.last_step = step

        for group, params in zip(self.optimizer.param_groups, self.warmup_params):
            omega = self.warmup_factor(step, **params)
            group['lr'] *= omega

    @contextmanager
    def dampening(self):
        for group, lr in zip(self.optimizer.param_groups, self.lrs):
            group['lr'] = lr
        yield
        self.lrs = [group['lr'] for group in self.optimizer.param_groups]
        self.dampen()

    def warmup_factor(self, step, **params):
        raise NotImplementedError


@SCHEDULERS.register_module()
class LinearWarmup(BaseWarmup):
    """Linear warmup schedule.

    Arguments:
        optimizer (Optimizer): an instance of a subclass of Optimizer
        period (int or list): Warmup period
        last_step (int): The index of last step. (Default: -1)
    """

    def __init__(self, optimizer, period, last_step=-1):
        group_count = len(optimizer.param_groups)
        warmup_params = get_warmup_params(period, group_count)
        super(LinearWarmup, self).__init__(optimizer, warmup_params, last_step)

    def warmup_factor(self, step, period):
        return min(1.0, (step+1) / period)


@SCHEDULERS.register_module()
class ExponentialWarmup(BaseWarmup):
    """Exponential warmup schedule.

    Arguments:
        optimizer (Optimizer): an instance of a subclass of Optimizer
        period (int or list): Effective warmup period
        last_step (int): The index of last step. (Default: -1)
    """

    def __init__(self, optimizer, period, last_step=-1):
        group_count = len(optimizer.param_groups)
        warmup_params = get_warmup_params(period, group_count)
        super(ExponentialWarmup, self).__init__(optimizer, warmup_params, last_step)

    def warmup_factor(self, step, period):
        return 1.0 - math.exp(-(step+1) / period)
