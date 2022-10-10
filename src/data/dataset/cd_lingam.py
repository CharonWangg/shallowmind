import os
import torch
import numpy as np
import scipy.io as scio
from ..builder import DATASETS, build_sampler
from ..pipeline import Compose
import causaldag as cd
from copy import deepcopy
from tqdm import tqdm


@DATASETS.register_module()
class CDLiNGAM(torch.utils.data.Dataset):
    def __init__(self, data_root='.cache', online=True, n_samples=5000, time_steps=10,
                      adj_cfg=dict(markov_class=False, n_nodes=10, density=0.2, low=0.5, high=2.0, allow_negative=True),
                      noise_cfg=dict(gaussian=True, scale=1.0, permutate=True),
                 to_corr=False, to_binary=True, sampler=None, pipeline=None):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.subject = 'LiNGAM'
        self.pipeline = Compose(pipeline)
        print('data_root')
        if not os.path.exists(data_root):
            os.makedirs(data_root)
            print(f'Creating data root: {data_root}')
        if online:
            self.generate_tons_of_data()
        else:
            self.xss = np.load(os.path.join(data_root, 'xss.npy'), mmap_mode='r')
            self.adjs = np.load(os.path.join(data_root, 'adjs.npy'), mmap_mode='r')
        if sampler is not None:
            self.data_sampler = getattr(torch.utils.data, sampler)(self)
        else:
            self.data_sampler = torch.utils.data.RandomSampler(self)

    def generate_adjacency_matrix(self):
        n_nodes, density, low, high = self.adj_cfg['n_nodes'], self.adj_cfg['density'], self.adj_cfg['low'], self.adj_cfg['high']
        if isinstance(density, (list, tuple)):
            density = np.random.uniform(density[0], density[1])
        prob_graph = np.float32(np.random.rand(n_nodes, n_nodes) < density)
        prob_graph = np.tril(prob_graph, -1)
        weight = np.round(np.random.uniform(low=low, high=high, size=[n_nodes, n_nodes]), 1)
        if not self.adj_cfg['allow_negative']:
            weight[np.random.randn(n_nodes, n_nodes) < 0] *= -1
        adj = (prob_graph != 0).astype(float) * weight
        return adj

    def generate_data(self, adj):
        gaussian, scale, permutate = self.noise_cfg['gaussian'], self.noise_cfg['scale'], self.noise_cfg['permutate']
        mean = np.zeros(adj.shape[0])
        if isinstance(scale, (list, tuple)):
            scale = np.random.uniform(scale[0], scale[1])
        if not gaussian:
            # LiNGAM (Non-Gaussian noise)
            # Nonlinearity exponent, selected to lie in [0.5, 0.8] or [1.2, 2.0].
            # (<1 gives subgaussian, >1 gives supergaussian)
            # Based on ICA-LiNGAM codes.
            # https://github.com/cdt15/lingam
            q = np.random.rand(sadj.shape[0]) * 1.1 + 0.5
            ixs = np.where(q > 0.8)
            q[ixs] = q[ixs] + 0.4

            # Generates disturbance variables
            ss = np.random.randn(self.time_steps, adj.shape[0])
            ss = np.sign(ss) * (np.abs(ss) ** q)

            # Normalizes the disturbance variables to have the appropriate scales
            ss = ss / np.std(ss, axis=0) * scale

        else:
            # Gaussian noise
            ss = np.random.randn(self.time_steps, adj.shape[0]) * scale

            # Generate the data one component at a time
        xs = np.zeros((self.time_steps, adj.shape[0]))
        for i in range(adj.shape[0]):
            # NOTE: columns of xs and ss correspond to rows of b
            xs[:, i] = ss[:, i] + xs.dot(adj[i, :]) + mean[i]

        if self.adj_cfg.get('markov_class', False):
            adj, node_list = self.dag2markovclass(adj).to_amat()
        if self.to_corr:
            xs = self.seq2corr(xs)

        if self.to_binary:
            adj = (adj != 0).astype(float)

        # Permute variables
        adj_ = deepcopy(adj)
        mean_ = deepcopy(mean)
        if permutate:
            p = np.random.permutation(adj.shape[0])
            xs[:, :] = xs[:, p]
            adj_[:, :] = adj_[p, :]
            adj_[:, :] = adj_[:, p]
            mean_[:] = mean_[p]

        return xs, adj_, mean_

    def generate_tons_of_data(self):
        # Generate tons of data
        self.xss = []
        self.adjs = []
        for i in tqdm(range(self.n_samples), desc='Generating data...'):
            adj = self.generate_adjacency_matrix()
            xs, adj_, mean_ = self.generate_data(adj)
            self.xss.append(xs)
            self.adjs.append(adj_)

        if self.online:
            # save data to data_root for next time offline use
            np.save(os.path.join(self.data_root, 'xss.npy'), np.stack(self.xss))
            np.save(os.path.join(self.data_root, 'adjs.npy'), np.stack(self.adjs))

    def dag2markovclass(self, dag):
        # convert DAG to Markov Class
        dag = cd.DAG.from_amat(dag)
        cpdag = dag.cpdag()
        return cpdag

    def seq2corr(self, xs):
        # convert sequence to correlation matrix
        corr = np.corrcoef(xs, rowvar=False)
        return corr

    def __getitem__(self, idx):
        if self.to_corr:
            seq = np.expand_dims(self.xss[idx], axis=0) if self.xss[idx].ndim == 2 else self.xss[idx]
        seq, label = {'seq': seq}, self.adjs[idx].reshape(-1)
        seq = self.pipeline(seq)
        label = torch.tensor(label, dtype=torch.int64)
        return seq, label

    def __len__(self):
        return self.n_samples
