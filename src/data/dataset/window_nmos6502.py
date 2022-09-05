import os
import cv2
import torch
import numpy as np
import pandas as pd
import pickle
from ..builder import DATASETS, build_sampler
from ..pipeline import Compose
from neuralpredictors.data.datasets import StaticImageSet, FileTreeDataset

@DATASETS.register_module()
class WindowNMOS6502(torch.utils.data.Dataset):
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

        # Load window data pkl
        self.ts_pkl = pickle.load(open(self.split, "rb"))

    def feature_mining(self, seq):
        # add shift to the sequence
        df = pd.DataFrame({'cause': seq[:, 0], 'effect': seq[:, 1]})
        df['cause_shift_1'] = df['cause'].shift(1)
        df['cause_shift_back_1'] = df['cause'].shift(-1)
        df['cause_shift_2'] = df['cause'].shift(2)
        df['cause_shift_back_2'] = df['cause'].shift(-2)
        df['cause_shift_3'] = df['cause'].shift(3)
        df['cause_shift_back_3'] = df['cause'].shift(-3)
        df['cause_shift_4'] = df['cause'].shift(4)
        df['cause_shift_back_4'] = df['cause'].shift(-4)
        df['cause_effect_diff'] = df['cause'] - df['effect']
        df['cause_shift_1_effect_diff'] = df['cause_shift_1'] - df['effect']
        df['cause_shift_back_1_effect_diff'] = df['cause_shift_back_1'] - df['effect']
        df['cause_shift_2_effect_diff'] = df['cause_shift_2'] - df['effect']
        df['cause_shift_back_2_effect_diff'] = df['cause_shift_back_2'] - df['effect']
        df['cause_shift_3_effect_diff'] = df['cause_shift_3'] - df['effect']
        df['cause_shift_back_3_effect_diff'] = df['cause_shift_back_3'] - df['effect']
        df['cause_shift_4_effect_diff'] = df['cause_shift_4'] - df['effect']
        df['cause_shift_back_4_effect_diff'] = df['cause_shift_back_4'] - df['effect']
        return df.fillna(0).to_numpy()

    def __getitem__(self, idx):
        seq = {'seq': np.array(self.ts_pkl[idx]['sample'], dtype=np.float32)}
        # seq = self.feature_mining(seq)

        label = torch.tensor(self.ts_pkl[idx]["label"], dtype=torch.int64)

        return self.pipeline(seq), label

    def __len__(self):
        return len(self.ts_pkl)
