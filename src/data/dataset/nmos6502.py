import torch
import numpy as np
import pandas as pd
from ..builder import DATASETS, build_sampler
from ..pipeline import Compose

@DATASETS.register_module()
class NMOS6502(torch.utils.data.Dataset):
    def __init__(self, data_root=None, split=None, interval=10, sampler=None, pipeline=None):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.subject = data_root.split('/')[-1].strip('.npy')
        self.pipeline = Compose(pipeline)
        self.check_files()
        self.interval = interval
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
        seq = {'seq': np.stack([self.seqs[int(self.time_series_df.iloc[idx]["transistor_1"])],
                                self.seqs[int(self.time_series_df.iloc[idx]["transistor_2"])]],
                               axis=-1).astype(np.float32)[::self.interval, :]}

        seq = self.pipeline(seq)
        # seq = self.feature_mining(seq)

        label = torch.tensor(self.time_series_df.iloc[idx]["label"], dtype=torch.int64)


        return seq, label

    def __len__(self):
        return len(self.time_series_df)
