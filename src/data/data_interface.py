from copy import deepcopy
import pytorch_lightning as pl
import torch.utils.data
from torch.utils.data import DataLoader
from shallowmind.src.data.builder import build_dataset, DATASETS
from shallowmind.src.data.utils import MaxCycleLoader

class DataInterface(pl.LightningDataModule):

    def __init__(self, data):
        super().__init__()
        self.save_hyperparameters()
        self.data_cfg = deepcopy(data)

    def get_multipe_dataset(self, dataset_cfg):
        if isinstance(dataset_cfg, list):
            dataset_cfg = [{k: v for k, v in d.items() if k not in ['multiple', 'multiple_key']} for d in dataset_cfg]
            datasets = [build_dataset(cfg) for cfg in dataset_cfg]
        elif dataset_cfg.pop('multiple', False):
            datasets = []
            multiple_key = dataset_cfg.pop('multiple_key', 'feature_dir')
            if isinstance(dataset_cfg[multiple_key], list):
                for key in dataset_cfg[multiple_key]:
                    cft_copy = dataset_cfg.copy()
                    cft_copy[multiple_key] = key
                    datasets.append(build_dataset(cft_copy))
            else:
                return build_dataset(dataset_cfg)
        else:
            dataset_cfg.pop('multiple_key', None)
            return build_dataset(dataset_cfg)
        return datasets

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            # prevent datasets from being multiply created
            if getattr(self, 'trainset', None) is None:
                self.trainset = self.get_multipe_dataset(self.data_cfg.train)
            if getattr(self, 'valset', None) is None:
                self.valset = self.get_multipe_dataset(self.data_cfg.val)
        if stage == 'val' or stage is None:
            self.valset = self.get_multipe_dataset(self.data_cfg.val)
        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.testset = self.get_multipe_dataset(self.data_cfg.test)

    def train_dataloader(self):
        if isinstance(self.trainset, list):
            dataloaders =  {trainset.subject:
                        DataLoader(trainset, batch_size=self.data_cfg.train_batch_size, sampler=trainset.data_sampler,
                                   num_workers=self.data_cfg.num_workers, pin_memory=True) for trainset in self.trainset}
            return MaxCycleLoader(dataloaders)
        else:
            return DataLoader(self.trainset, batch_size=self.data_cfg.train_batch_size, sampler=self.trainset.data_sampler,
                              num_workers=self.data_cfg.num_workers, pin_memory=True)

    def val_dataloader(self):
        if isinstance(self.valset, list):
            dataloaders =  {valset.subject:
                        DataLoader(valset, batch_size=self.data_cfg.val_batch_size, sampler=valset.data_sampler,
                                   num_workers=self.data_cfg.num_workers, shuffle=False, pin_memory=True) for valset in self.valset}
            return MaxCycleLoader(dataloaders)
        else:
            return DataLoader(self.valset, batch_size=self.data_cfg.val_batch_size, sampler=self.valset.data_sampler,
                              num_workers=self.data_cfg.num_workers, shuffle=False, pin_memory=True)

    def test_dataloader(self):
        if isinstance(self.testset, list):
            dataloaders =  {testset.subject:
                        DataLoader(testset, batch_size=self.data_cfg.test_batch_size, sampler=testset.data_sampler,
                                   num_workers=self.data_cfg.num_workers, shuffle=False, pin_memory=True) for testset in self.testset}
            return MaxCycleLoader(dataloaders)
        else:
            return DataLoader(self.testset, batch_size=self.data_cfg.test_batch_size, sampler=self.testset.data_sampler,
                              num_workers=self.data_cfg.num_workers, shuffle=False, pin_memory=True)

