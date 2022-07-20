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
    def __init__(self, data_root=None, tier='train', stack=False, file_tree=True, feature_dir='images',
                 data_keys=['images', 'responses', 'behavior', 'pupil_center', 'frame_image_id'],
                 label_smooth=None, sampler=None, pipeline=None):
        assert os.path.exists(data_root), f"data_root {data_root} does not exist"
        self.data_root = data_root
        if feature_dir not in os.listdir(data_root):
            raise ValueError(f"feature_dir {feature_dir} does not exist")
        else:
            self.feature_dir = os.path.join(data_root, feature_dir)
        self.tier = tier
        self.stack = stack
        self.file_tree = file_tree
        self.data_keys = data_keys
        self.label_smooth = label_smooth
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
        if not self.stack and self.tier == 'train':
            self.data_sampler = SubsetRandomSampler(subset_idex)
        else:
            self.data_sampler = SubsetSequentialSampler(subset_idex)
            if self.stack:
                self.subset_idex = subset_idex


        return dataset

    def __getitem__(self, idx):
        data = self.dataset[idx]
        # rename keys to match the convention of the dataset
        data['image'] = data.pop('images')
        data = self.pipeline(data)
        data['images'] = data.pop('image')

        # label smoothing
        if self.label_smooth is not None and self.tier == 'train':
            data['responses'] = np.where(data['responses'] < self.label_smooth, 0.0, data['responses'])

        if self.stack:
            current_idx = np.where(self.subset_idex==idx)[0]
            last_idx = self.subset_idex[current_idx-1] if current_idx > 0 else idx
            next_idx = self.subset_idex[current_idx+1] if current_idx < len(self.subset_idex)-1 else idx
            last_image = self.dataset[int(last_idx)]
            next_image = self.dataset[int(next_idx)]
            last_image['image'] = last_image.pop('images')
            next_image['image'] = next_image.pop('images')
            last_image = self.pipeline(last_image)
            next_image = self.pipeline(next_image)
            last_image['images'] = last_image.pop('image')
            next_image['images'] = next_image.pop('image')

            data['images'] = np.stack([last_image['images'][-1],
                                             data['images'][-1],
                                             next_image['images'][-1]], axis=0)


        return data, data['responses']


    def __len__(self):
        return len(self.dataset)
