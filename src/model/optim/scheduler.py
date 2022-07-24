import torch
from ..builder import SCHEDULERS

@SCHEDULERS.register_module()
class Constant(torch.optim.lr_scheduler.ConstantLR):
    '''fix lr scheduler'''
    def __init__(self, optimizer, max_epochs=None, max_steps=None, factor=1.0, total_iters=5, last_epoch=-1, verbose=False):
        super(Constant, self).__init__(optimizer=optimizer, factor=factor, total_iters=max_epochs, last_epoch=last_epoch, verbose=verbose)

@SCHEDULERS.register_module()
class Step(torch.optim.lr_scheduler.StepLR):
    '''step lr scheduler'''
    def __init__(self, optimizer, step_size, max_epochs=None, max_steps=None, gamma=0.1, last_epoch=-1, verbose=False):
        super(Step, self).__init__(optimizer=optimizer, step_size=step_size, gamma=gamma, last_epoch=last_epoch, verbose=verbose)

@SCHEDULERS.register_module()
class MultiStep(torch.optim.lr_scheduler.MultiStepLR):
    '''step lr scheduler'''
    def __init__(self, optimizer, milestones, max_epochs=None, max_steps=None, gamma=0.1, last_epoch=-1, verbose=False):
        super(MultiStep, self).__init__(optimizer=optimizer, milestones=milestones, gamma=gamma, last_epoch=last_epoch, verbose=verbose)

@SCHEDULERS.register_module()
class Linear(torch.optim.lr_scheduler.LinearLR):
    '''linear lr scheduler'''
    def __init__(self, optimizer, start_factor=1.0 / 3, end_factor=1.0,
                       max_epochs=5, max_steps=None, last_epoch=-1, verbose=False):
        super(Linear, self).__init__(optimizer=optimizer, start_factor=start_factor, end_factor=end_factor,
                                     total_iters=max_epochs, last_epoch=last_epoch, verbose=verbose)

@SCHEDULERS.register_module()
class CosineAnnealing(torch.optim.lr_scheduler.CosineAnnealingLR):
    '''cosine annealing lr scheduler'''
    def __init__(self, optimizer, max_epochs=None, max_steps=None, min_lr=0, last_epoch=-1, verbose=False):
        super(CosineAnnealing, self).__init__(optimizer=optimizer, T_max=max_steps, eta_min=min_lr,
                                                                   last_epoch=last_epoch, verbose=verbose)

@SCHEDULERS.register_module()
class OneCycle(torch.optim.lr_scheduler.OneCycleLR):
    '''one cycle annealing lr scheduler'''
    def __init__(self, optimizer, max_lr=0.1, max_steps=-1, max_epochs=-1, pct_start=0.3, anneal_strategy='cos',
                       cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=25., final_div_factor=1e4,
                       three_phase=False, last_epoch=-1, verbose=False):
        steps_per_epoch = max_steps // max_epochs + 1
        super(OneCycle, self).__init__(optimizer=optimizer, max_lr=max_lr,
                                       total_steps=max_steps, steps_per_epoch=steps_per_epoch, epochs=max_epochs,
                                       pct_start = pct_start, anneal_strategy = anneal_strategy,
                                       cycle_momentum=cycle_momentum, base_momentum=base_momentum, max_momentum=max_momentum,
                                       div_factor=div_factor, final_div_factor=final_div_factor,
                                       three_phase=three_phase, last_epoch=last_epoch, verbose=verbose)

@SCHEDULERS.register_module()
class ReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    '''reduce lr on plateau scheduler'''
    def __init__(self, optimizer, max_epochs=-1, max_steps=-1, mode='min', factor=0.1, patience=10, verbose=False,
                 threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08):
        super(ReduceLROnPlateau, self).__init__(optimizer=optimizer, mode=mode, factor=factor, patience=patience, verbose=verbose,
                                                threshold=threshold, threshold_mode=threshold_mode, cooldown=cooldown, min_lr=min_lr,
                                                eps=eps)











