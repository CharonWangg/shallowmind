import torch
import numpy as np
import scipy.io as scio
from ..builder import DATASETS, build_sampler
from ..pipeline import Compose


@DATASETS.register_module()
class NetSim(torch.utils.data.Dataset):
    def __init__(self, data_root=None, split=None, interval=10, sampler=None, pipeline=None):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.check_files()
        self.subject = data_root.split('/')[-1].strip('.mat')
        self.pipeline = Compose(pipeline)
        self.dataset = self.check_files()
        self.interval = interval
        if sampler is not None:
            self.data_sampler = getattr(torch.utils.data, sampler)(self)
        else:
            self.data_sampler = torch.utils.data.RandomSampler(self)

    def check_files(self):
        if self.data_root is None:
            raise Exception("Invalid dataset path")

        # Load data
        data = scio.loadmat(self.data_root)
        self.n_subjects, self.n_node, self.duration = int(data['Nsubjects']), int(data['Nnodes']), int(data['Ntimepoints'])
        self.seqs = data['ts']
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
        if seq.shape[0] > 500:
            seq = seq[::10]
        # padding
        seq = {'seq': np.pad(seq, pad_width=[(0, 500 - seq.shape[0]), (0, 0)], mode='constant', constant_values=0)}

        seq = self.pipeline(seq)
        label = torch.tensor(example['label'], dtype=torch.int64)
        return seq, label

    def __len__(self):
        return len(self.examples)
