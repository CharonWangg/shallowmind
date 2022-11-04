from collections import deque
import numpy as np

def cycle(iterable):
    # cycle through an iterable
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


class MaxCycleLoader:
    """
    Cycles through loaders until the loader with largest size
    """
    def __init__(self, loaders):
        self.loaders = loaders
        self.max_batches = max([len(loader) for loader in self.loaders.values()])

    def __iter__(self):
        cycles = [cycle(loader) for loader in self.loaders.values()]
        for k, loader, _ in zip(
            cycle(self.loaders.keys()),
            (cycle(cycles)),
            range(len(self.loaders) * self.max_batches),
        ):
            yield next(loader)

    def __len__(self):
        return len(self.loaders) * self.max_batches


class ReplayBuffer:
    '''
    Replay Buffer for storing past experiences
    '''
    def __init__(self, buffer_size=1000, field_dtype=None):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        if field_dtype is not None:
            self.field_dtype = field_dtype

    def append(self, data):
        self.buffer.append(data)

    def sample(self, sample_size):
        if len(self.buffer) == 0:
            raise ValueError('Buffer is empty, can not sample')
        indices = np.random.choice(len(self.buffer), sample_size, replace=False)
        # ['state', 'action', 'reward', 'done', 'new_state']
        batch_data = [{k: np.array(v, dtype=self.field_dtype[k])
                       for k, v in self.buffer[i].items()}
                      for i in indices]
        return batch_data
