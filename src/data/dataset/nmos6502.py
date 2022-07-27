import os
import cv2
import torch
import numpy as np
import pandas as pd
from ..builder import DATASETS, build_sampler
from ..pipeline import Compose
from neuralpredictors.data.datasets import StaticImageSet, FileTreeDataset

@DATASETS.register_module()
class NMOS6502(torch.utils.data.Dataset):
    def __init__(self, data_root=None, split=None, sampler=None, pipeline=None):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.check_files()
        self.pipeline = Compose(pipeline)
        self.dataset = self.check_files()
        if sampler is not None:
            self.data_sampler = getattr(torch.utils.data, sampler)(self)
        else:
            self.data_sampler = torch.utils.data.RandomSampler(self)

    def check_files(self):
        if self.data_root is None:
            raise Exception("Invalid dataset path")

        # Load data
        self.time_series_df = pd.read_csv(self.split)
        self.ts_df = pd.read_csv('/home/charon/project/nmos_inference/raw_data/transistors.csv')
        self.seqs = np.load(self.data_root, mmap_mode='r')

    def __getitem__(self, idx):
        seq = torch.stack((torch.tensor(self.seqs[int(self.time_series_df.iloc[idx]["transistor_1"])], dtype=torch.float32),
                            torch.tensor(self.seqs[int(self.time_series_df.iloc[idx]["transistor_2"])], dtype=torch.float32)),
                            dim=1)
        label = torch.tensor(self.time_series_df.iloc[idx]["label"], dtype=torch.int64)

        return self.pipeline(seq), label

    def __len__(self):
        return len(self.time_series_df)
