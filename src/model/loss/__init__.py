from .poisson_loss import PoissonLoss, PoissonLikeGaussianLoss
from .torch_loss import TorchLoss
from .smp_loss import SMPLoss
from .focal_loss import FocalLoss
from .correlation_loss import CorrelationLoss
from .kl_divergence_loss import KLDivergenceLoss

__all__ = ['PoissonLoss', 'PoissonLikeGaussianLoss', 'TorchLoss', 'SMPLoss', 'FocalLoss', 'CorrelationLoss',
           'KLDivergenceLoss']