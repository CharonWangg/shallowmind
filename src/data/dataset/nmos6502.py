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
        self.seqs = np.load(self.data_root, mmap_mode='r')

    def __getitem__(self, idx):
        seq = {'seq': np.stack([self.seqs[int(self.time_series_df.iloc[idx]["transistor_1"])],
                                self.seqs[int(self.time_series_df.iloc[idx]["transistor_2"])]],
                               axis=-1).astype(np.float32)[::self.interval, :]}

        seq = self.pipeline(seq)
        label = torch.tensor(self.time_series_df.iloc[idx]["label"], dtype=torch.int64)
        return seq, label

    def __len__(self):
        return len(self.time_series_df)
