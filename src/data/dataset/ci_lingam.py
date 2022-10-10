import os
import torch
import numpy as np
import pandas as pd
import scipy.io as scio
from ..builder import DATASETS, build_sampler
from ..pipeline import Compose
import causaldag as cd
from copy import deepcopy
from tqdm import tqdm
from joblib import Parallel, delayed


@DATASETS.register_module()
class CILiNGAM(torch.utils.data.Dataset):
    def __init__(self, data_root='.cache', online=True, base_seed=42, mini_batch_size=500, n_datasets=100,
                 n_samples=[500, 10000],
                 variable_cfg={'iv_strength': [], 'conf_strength': [], 'treat_effect': [], 'conf_effect': []},
                 sampler=None, pipeline=None):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.subject = 'CILiNGAM'
        self.pipeline = Compose(pipeline)
        if not os.path.exists(data_root):
            os.makedirs(data_root)
            print(f'Creating data root: {data_root}')

        # configure the parameters in graph model
        self.configure_graph_params()
        self.generate_tons_of_data()
        self.mini_batch2batch()

        if sampler is not None:
            self.data_sampler = getattr(torch.utils.data, sampler)(self)
        else:
            self.data_sampler = torch.utils.data.RandomSampler(self)

    def configure_graph_params(self):
        self.graph_params = []
        for j in range(self.n_datasets):
            seed = j + int(self.base_seed)
            np.random.seed(seed)
            # Generate sample params
            params = {'n_samples': int(np.random.uniform(self.n_samples[0], self.n_samples[1])), 'seed': seed}
            params.update({k: np.random.uniform(v[0], v[1]) if len(v) != 0 else np.random.uniform(0, 1)
                           for k, v in self.variable_cfg.items()})
            self.graph_params.append(params)

    def generate_const_linear_iv(self, graph_params):
        """
        Generates linear IV with constant treatment effects.

        Args:
            n_samples (int): num samples to generate
            seed (int): seed for reproducibilty
            pi (float): instrument "strength"
            psi (float): confounding "strength"
            tau (float): treatment effect
            gamma (float): confound effect

        Returns:
            pd.DataFrame
        """
        pi, psi, tau, gamma, seed, n_samples = graph_params['iv_strength'], graph_params['conf_strength'], \
                                               graph_params['treat_effect'], graph_params['conf_effect'], \
                                               graph_params['seed'], graph_params['n_samples']
        np.random.seed(seed),
        Z = np.random.normal(0, 1, size=n_samples)  # np.random.uniform(0, 10, n_samples)
        C = np.random.normal(0, 1, size=n_samples)  # np.random.uniform(0, 10, n_samples)
        eta = np.random.normal(0, 1, size=n_samples)
        const = np.random.uniform(-1, 1)

        T = const + (pi * Z) + (psi * C) + eta

        epsilon = np.random.normal(0, 1, size=n_samples)
        beta = np.random.uniform(-1, 1)

        Y = beta + (tau * T) + (gamma * C) + epsilon

        data = np.concatenate([Z.reshape(-1, 1),
                               C.reshape(-1, 1),
                               T.reshape(-1, 1),
                               Y.reshape(-1, 1), ],
                              axis=1)

        df = pd.DataFrame(data, columns=['Z', 'C', 'T', 'Y'])
        df['n_samples'] = [n_samples] * n_samples
        # df['seed'] = [seed] * n_samples
        # df['pi'] = [pi] * n_samples
        # df['psi'] = [psi] * n_samples
        # df['tau'] = [tau] * n_samples
        # df['gamma'] = [gamma] * n_samples

        return df

    def generate_tons_of_data(self):
        if self.online:
            # Generate tons of data
            def generate_data(i, params):
                df = self.generate_const_linear_iv(params)
                df['dataset_id'] = [i] * len(df)
                df['tau'] = [params['treat_effect']] * len(df)
                return df, params['treat_effect']

            self.dataset = [generate_data(i, params)
                            for i, params in
                            tqdm(enumerate(self.graph_params), total=len(self.graph_params), desc='Generating data...')]
            # pd.concat([df for df, _ in self.dataset], axis=0).to_csv(
            #     os.path.join(self.data_root, f'{self.subject}-n_datasets={self.n_datasets}-'
            #                                  f'n_samples={self.n_samples}-variable_cfg={self.variable_cfg}.csv'),
            #     index=False)
        else:
            if not os.path.exists(self.data_root):
                raise FileNotFoundError(f'Cannot find data root: {self.data_root}')
            self.dataset = pd.read_csv(self.data_root)
            self.dataset = [(group, group['tau'].values()[0])
                            for i, group in self.dataset.groupby(by='dataset_id')]

        self.dataset = [(dataset.drop(columns=['C', 'tau', 'dataset_id']).to_numpy(), tau) for dataset, tau in
                        self.dataset]
        # decompose the mini-batch into samples
        # self.dataset = [(sample, tau) for dataset, tau in self.dataset for sample in dataset]

    def mini_batch2batch(self):
        # sample mini-batch from each dataset and concatenate them to a whole batch
        def real_random_sample(_seed, _n_samples):
            _seed += self.base_seed
            np.random.seed(_seed)
            if _n_samples > self.mini_batch_size:
                return np.random.choice(_n_samples, self.mini_batch_size, replace=False)
            else:
                return np.random.choice(_n_samples, self.mini_batch_size, replace=True)

        n_mini_batches = int(max(self.n_samples) // self.mini_batch_size)
        self.dataset = sum([[(dataset[real_random_sample(j * 10, len(dataset))], tau)
                             for j, (dataset, tau) in enumerate(self.dataset)]
                            for i in tqdm(range(n_mini_batches), desc='Converting to batch...')], [])

    def feature_exaction(self, x):
        data = deepcopy(x[:, :-1])

        cov_mat = np.cov(data, rowvar=False)

        var = np.diag(cov_mat).reshape(-1)
        var = var[var != 0]

        cov = np.array([cov_mat[0, 1], cov_mat[0, 2], cov_mat[1, 2]])
        mean = np.mean(data, axis=0)

        x = np.concatenate([mean, var, cov, [x[0, -1]]], axis=0)

        return x

    def __getitem__(self, idx):
        # dataset: [batch_size, mini_batch_size, n_variables]
        seq, label = self.dataset[idx]
        seq, label = {'seq': seq}, torch.tensor(label, dtype=torch.float32)

        seq = self.pipeline(seq)
        return seq, label

    def __len__(self):
        return len(self.dataset)
