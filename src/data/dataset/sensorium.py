import os
import cv2
import copy
import torch
from torch.utils.data import ConcatDataset
import numpy as np
from ..builder import DATASETS, build_sampler
from ..pipeline import Compose
from neuralpredictors.data.datasets import StaticImageSet, FileTreeDataset
from torch.utils.data.sampler import SubsetRandomSampler
from neuralpredictors.data.samplers import SubsetSequentialSampler
from neuralpredictors.data.transforms import NeuroNormalizer, ScaleInputs

@DATASETS.register_module()
class Sensorium(torch.utils.data.Dataset):
    def __init__(self, data_root=None, tier='train', file_tree=True, feature_dir='images', per_neuron=False,
                 data_keys=['images', 'responses', 'behavior', 'pupil_center', 'frame_image_id'],
                 label_smooth=None, pipeline=None):
        assert os.path.exists(data_root), f"data_root {data_root} does not exist"
        self.data_root = data_root
        if feature_dir not in os.listdir(data_root):
            raise ValueError(f"feature_dir {feature_dir} does not exist")
        else:
            self.feature_dir = os.path.join(data_root, feature_dir)

        self.tier = tier
        self.file_tree = file_tree
        self.data_keys = data_keys
        self.per_neuron = per_neuron
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
        # flatten the dataset if per_neuron is True
        if self.per_neuron:
            self.n_neurons = len(dataset[0]['responses'])
            self.num_units = len(dataset) * len(dataset[0]['responses'])
            self.cell_motor_coordinates = dataset.neurons.cell_motor_coordinates
            subset_idex = np.where(dataset.trial_info.tiers == self.tier)[0]
            subset_idex = np.concatenate([np.linspace(i, i + self.n_neurons) for i in subset_idex])
        else:
            # acquire trainset/valdset/testset by tier array
            tier_array = dataset.trial_info.tiers
            subset_idex = np.where(tier_array == self.tier)[0]
        if self.tier == 'train':
            self.data_sampler = SubsetRandomSampler(subset_idex)
        else:
            self.data_sampler = SubsetSequentialSampler(subset_idex)

        return dataset

    def __getitem__(self, idx):
        if not self.per_neuron:
            data = self.dataset[idx]
            # rename keys to match the convention of the dataset
            data['image'] = data.pop('images')
            data = self.pipeline(data)
            data['images'] = data.pop('image')

            # label smoothing
            if self.label_smooth is not None and self.tier == 'train':
                data['responses'] = np.where(data['responses'] < self.label_smooth, 0.0, data['responses'])

            # add subject to sample
            data['subject'] = self.subject

            return data, data['responses']
        else:
            sample_idx = int(idx // self.n_neurons)
            unit_idx = int(idx % self.n_neurons)
            data = self.dataset[sample_idx]
            data['unit_idx'] = unit_idx
            data['responses'] = data['responses'][unit_idx]
            data['cell_motor_coordinates'] = self.cell_motor_coordinates[unit_idx]

            # rename keys to match the convention of the dataset
            data['image'] = data.pop('images')
            data = self.pipeline(data)
            data['images'] = data.pop('image')

            # label smoothing
            if self.label_smooth is not None and self.tier == 'train':
                data['responses'] = np.where(data['responses'] < self.label_smooth, 0.0, data['responses'])

            # add subject to sample
            data['subject'] = self.subject
            return data, data['responses']

    def __len__(self):
        if self.per_neuron:
            return len(self.dataset)
        else:
            return len(self.num_units)
