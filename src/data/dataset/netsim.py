import torch
import numpy as np
import scipy.io as scio
from ..builder import DATASETS, build_sampler
from ..pipeline import Compose
from copy import deepcopy


@DATASETS.register_module()
class NetSim(torch.utils.data.Dataset):
    def __init__(self, data_root=None, split=None, percentage=1.0, interval=10, max_length=200, sampler=None, pipeline=None):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.subject = data_root.split('/')[-1].strip('.mat')
        self.pipeline = Compose(pipeline)
        self.check_files()
        if sampler is not None:
            self.data_sampler = getattr(torch.utils.data, sampler)(self)
        else:
            self.data_sampler = torch.utils.data.RandomSampler(self)

    def check_files(self):
        if isinstance(self.percentage, float):
            self.percentage = [0, self.percentage]

        if self.data_root is None:
            raise Exception("Invalid dataset path")

        # Load data
        data = scio.loadmat(self.data_root)
        self.n_subjects, self.n_node, self.duration = int(data['Nsubjects']), int(data['Nnodes']), int(data['Ntimepoints'])
        slice = [int(self.percentage[0]*self.n_subjects), int(self.percentage[1]*self.n_subjects)]
        data['ts'] = data['ts'][self.duration*slice[0]:self.duration*slice[1]]
        self.seqs = deepcopy(data['ts'])
        data['net'] = data['net'][slice[0]:slice[1]]
        self.examples = []
        for i, start in enumerate(range(0, data['ts'].shape[0], self.duration)):
            for j in range(self.n_node):
                for k in range(self.n_node):
                    if j == k: continue
                    self.examples.append({
                        'cause': j,
                        'effect': k,
                        'start_index': start,
                        'end_index': start+self.duration,
                        'label': 1 if data['net'][i][j, k] > 0 else 0
                    })

    def __getitem__(self, idx):
        example = self.examples[idx]
        seq = np.stack([self.seqs[example['start_index']:example['end_index'], example['cause']],
                        self.seqs[example['start_index']:example['end_index'], example['effect']]], axis=-1).astype(np.float32)

        # downsample
        if seq.shape[0] > self.max_length:
            seq = seq[::10]
        # padding
        seq = {'seq': np.pad(seq, pad_width=[(0, self.max_length - seq.shape[0]), (0, 0)], mode='constant', constant_values=0)}

        seq = self.pipeline(seq)
        label = torch.tensor(example['label'], dtype=torch.int64)
        return seq, label

    def __len__(self):
        return len(self.examples)
