import torch
import numpy as np
from ..builder import DATASETS
from tqdm import tqdm
import gym
from ..utils import ReplayBuffer


@DATASETS.register_module()
class Gym(torch.utils.data.dataset.IterableDataset):
    field_dtype = {'state': np.float32,
                  'action': np.float32,
                  'reward': np.float32,
                  'done': bool,
                  'new_state': np.float32}

    def __init__(self, env_name='CartPole-v0', buffer_cfg=dict(buffer_size=1000, field_dtype=None), sample_size=200):
        if buffer_cfg.get('field_dtype', None) is None:
            buffer_cfg.field_dtype = self.field_dtype
        self.env = gym.make(env_name)
        self.buffer = ReplayBuffer(**buffer_cfg)
        self.sample_size = sample_size
        self.populate()
        self.data_sampler = None

    def populate(self):
        # fill up the replay buffer with random actions
        state = self.env.reset()
        state = state[0] if isinstance(state, tuple) else state
        for _ in tqdm(range(self.buffer.buffer_size), desc='Populating Replay Buffer', leave=True):
            action = self.env.action_space.sample()
            new_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            exp = {'state': state, 'action': action, 'reward': reward, 'done': done, 'new_state': new_state}
            self.buffer.append(exp)
            state = new_state
            if done:
                state = self.env.reset()
                state = state[0] if isinstance(state, tuple) else state

    def __iter__(self):
        batch_data = self.buffer.sample(self.sample_size)
        for i in range(len(batch_data)):
            yield batch_data[i], 0

    def __len__(self):
        return self.sample_size