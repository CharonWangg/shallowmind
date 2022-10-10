import os
import torch
from ..builder import DATASETS, build_sampler
from ..pipeline import Compose
import torchvision


@DATASETS.register_module()
class TorchVision(torch.utils.data.Dataset):
    def __init__(self, dataset_name, data_root='.cache/', download=True, sampler=None, pipeline=None, **kwargs):
        if download == False and os.path.exists(data_root):
            self.dataset = getattr(torchvision.datasets, dataset_name)(root=data_root, download=False, **kwargs)
        else:
            self.dataset = getattr(torchvision.datasets, dataset_name)(root=data_root, download=True, **kwargs)
        self.subject = dataset_name
        self.pipeline = Compose(pipeline)
        if sampler is not None:
            self.data_sampler = getattr(torch.utils.data, sampler)(self)
        else:
            self.data_sampler = torch.utils.data.RandomSampler(self)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = self.pipeline({'image': img})
        label = torch.tensor(label)

        return img, label

    def __len__(self):
        return len(self.dataset)
