from ..builder import SAMPLERS
from torch.utils.data.sampler import SubsetRandomSampler
from neuralpredictors.data.samplers import SubsetSequentialSampler

@SAMPLERS.register_module()
class SubsetRandomSampler(SubsetRandomSampler):
    def __init__(self, idx, **kwargs):
        super(SubsetRandomSampler, self).__init__(idx, **kwargs)

@SAMPLERS.register_module()
class SubsetSequentialSampler(SubsetSequentialSampler):
    def __init__(self, idx, **kwargs):
        super(SubsetSequentialSampler, self).__init__(idx, **kwargs)
