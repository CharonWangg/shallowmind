import os
import torch
import pickle
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
from ..pipeline import Compose
from ..builder import DATASETS


@DATASETS.register_module()
class Friendship(torch.utils.data.Dataset):
    def __init__(self, data_root=None, prefix='final_eye_close', split=None, _5D=False,
                       n_subset=10, sampler=None, pipeline=None):
        self.__dict__.update(locals())
        if _5D and n_subset % 5 != 0:
            raise ValueError('n_subset must be divisible by 5')
        self.check_files()
        self.pipeline = Compose(pipeline)
        if sampler is not None:
            self.data_sampler = getattr(torch.utils.data, sampler)(self)
        else:
            self.data_sampler = torch.utils.data.RandomSampler(self)

    def check_files(self):
        self.df = pd.read_csv(self.split)
        subjects = set(list(self.df['subject1']) + list(self.df['subject2']))
        subjects_segments = [glob(os.path.join(self.data_root, f'{self.prefix}_{subject}_*.npy')) for subject in subjects]
        subjects_segments = {'_'.join(subject_segments[0].split('/')[-1].split('_')[:4]):
                                [subject_segments[i] for i in list(np.random.choice(len(subject_segments), self.n_subset))]
                                for subject_segments in subjects_segments}

        self.segments = [{'segment1': segment,
                         'segment2': segment2,
                         'label': row['label']}
                                                for idx, row in self.df.iterrows()
                                                for jdx, segment in enumerate(subjects_segments[f'{self.prefix}_{row["subject1"]}'])
                                                for segment2 in subjects_segments[f'{self.prefix}_{row["subject2"]}'][:jdx+1]]
        if self._5D:
            self.segments = [[self.segments[i*5+j] for j in range(5)] for i in range(0, len(self.segments), 5)]
        self.z_score = pickle.load(open(os.path.join(Path(self.data_root).parent, 'mean_std.pkl'), 'rb'))

    def load_normalize(self, segment):
        # load data and normalize to [0, 1]
        x = np.load(segment)
        x_name = int(segment.split('/')[-1].split('_')[3])
        x = (x - self.z_score[x_name]['min'].reshape(-1, 1)) / (self.z_score[x_name]['max'].reshape(-1, 1) -
                                                                self.z_score[x_name]['min'].reshape(-1, 1))
        return x

    def __getitem__(self, idx):
        row = self.segments[idx]
        if self._5D:
            row_5D = [(self.load_normalize(_row['segment1']), self.load_normalize(_row['segment2'])) for _row in row]
            x1, x2 = np.stack([x[0] for x in row_5D], axis=0), np.stack([x[1] for x in row_5D], axis=0)
        else:
            x1, x2 = self.load_normalize(row['segment1']), self.load_normalize(row['segment2'])

        seq = {'seq': np.stack([x1, x2], axis=-1)}
        seq = self.pipeline(seq)

        label = torch.tensor(row[2]['label'] if self._5D else row['label'], dtype=torch.int64)

        return seq, label

    def __len__(self):
        return len(self.segments)
