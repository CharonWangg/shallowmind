import os
import cv2
import torch
import numpy as np
from ..builder import DATASETS, build_sampler
from ..pipeline import Compose
from neuralpredictors.data.datasets import StaticImageSet, FileTreeDataset
from torch.utils.data.sampler import SubsetRandomSampler
from neuralpredictors.data.samplers import SubsetSequentialSampler
from neuralpredictors.data.transforms import NeuroNormalizer

@DATASETS.register_module()
class Sensorium:
    def __init__(self, data_root=None, tier='train', file_tree=True, feature_dir='images',
                 data_keys=['images', 'responses', 'behavior', 'pupil_center', 'frame_image_id'],
                 sampler=None, pipeline=None):
        assert os.path.exists(data_root), f"data_root {data_root} does not exist"
        self.data_root = data_root
        if feature_dir not in os.listdir(data_root):
            raise ValueError(f"feature_dir {feature_dir} does not exist")
        else:
            self.feature_dir = os.path.join(data_root, feature_dir)
        self.tier = tier
        self.file_tree = file_tree
        self.data_keys = data_keys
        if 'ColorImageNet' in feature_dir:
            self.subject = feature_dir.split("static")[-1].split("-ColorImageNet")[0]
        else:
            self.subject = feature_dir.split("static")[-1].split("-GrayImageNet")[0]
        self.dataset = self.check_files()
        self.pipeline = Compose(pipeline)
        self.dataset.transforms.extend([NeuroNormalizer(self.dataset, exclude=['frame_image_id'],
                                                        inputs_mean=None, inputs_std=None)])

    def check_files(self):
        if self.file_tree:
            dataset = FileTreeDataset(self.feature_dir, output_dict=True, *self.data_keys)
        else:
            dataset = StaticImageSet(self.feature_dir, *self.data_keys)

        # acquire trainset/valdset/testset by tier array
        tier_array = dataset.trial_info.tiers
        subset_idex = np.where(tier_array == self.tier)[0]
        self.data_sampler = (
            SubsetRandomSampler(subset_idex)
            if self.tier == "train"
            else SubsetSequentialSampler(subset_idex)
        )
        return dataset

    def __getitem__(self, idx):
        data = self.dataset[idx]
        # rename keys to match the convention of the dataset
        # convert greyscale to rgb
        data['image'] = data.pop('images')
        data = self.pipeline(data)
        data['images'] = data.pop('image')

        return data, data['responses']


    def __len__(self):
        return len(self.dataset)
