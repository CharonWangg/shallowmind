import math
from .base import BaseWarmup
from ....builder import SCHEDULERS


def rho_inf_fn(beta2):
    return 2.0 / (1 - beta2) - 1


def rho_fn(t, beta2, rho_inf):
    b2t = beta2 ** t
    rho_t = rho_inf - 2 * t * b2t / (1 - b2t)
    return rho_t


def get_offset(beta2, rho_inf):
    if not beta2 > 0.6:
        raise ValueError('beta2 ({}) must be greater than 0.6'.format(beta2))
    offset = 1
    while True:
        if rho_fn(offset, beta2, rho_inf) > 4:
            return offset
        offset += 1

@SCHEDULERS.register_module()
class RAdamWarmup(BaseWarmup):
    """RAdam warmup schedule"""

    def __init__(self, optimizer, last_step=-1):
        warmup_params = [
            dict(
                beta2=x['betas'][1],
                rho_inf=rho_inf_fn(x['betas'][1]),
            )
            for x in optimizer.param_groups
        ]
        for x in warmup_params:
            x['offset'] = get_offset(**x)
        super(RAdamWarmup, self).__init__(optimizer, warmup_params, last_step)

    def warmup_factor(self, step, beta2, rho_inf, offset):
        rho = rho_fn(step+offset, beta2, rho_inf)
        numerator = (rho - 4) * (rho - 2) * rho_inf
        denominator = (rho_inf - 4) * (rho_inf - 2) * rho
        return math.sqrt(numerator/denominator)
