import os
import torch
from ..builder import DATASETS, build_sampler
from ..pipeline import Compose
import torchvision


@DATASETS.register_module()
class DualImageNet(torch.utils.data.Dataset):
    def __init__(self, dataset_name='ImageNet', data_root='.cache/', download=True, sampler=None, pipeline=None, **kwargs):
        if download == False and os.path.exists(data_root):
            self.dataset = getattr(torchvision.datasets, dataset_name)(root=data_root, download=False, **kwargs)
        else:
            self.dataset = getattr(torchvision.datasets, dataset_name)(root=data_root, download=True, **kwargs)
        self.subject = dataset_name
        if isinstance(pipeline, dict):
            self.orig_pipeline = Compose(pipeline['orig_pipeline'])
            self.aug_pipeline = Compose(pipeline['aug_pipeline'])
        else:
            self.orig_pipeline = Compose(pipeline)
            self.aug_pipeline = Compose(pipeline)
        if sampler is not None:
            self.data_sampler = getattr(torch.utils.data, sampler)(self)
        else:
            self.data_sampler = torch.utils.data.RandomSampler(self)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # tricky here, image -> augmented image, orig_image -> original image
        orig_img = self.orig_pipeline({'image': img})
        aug_img = self.aug_pipeline({'image': img})
        label = torch.tensor(label)

        return {'orig_image': orig_img, 'aug_image': aug_img}, label

    def __len__(self):
        return len(self.dataset)
