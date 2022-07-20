import pytorch_lightning as pl
import torch.utils.data
from torch.utils.data import DataLoader
from shallowmind.src.data.builder import build_dataset, DATASETS

class DataInterface(pl.LightningDataModule):

    def __init__(self, data):
        super().__init__()
        self.save_hyperparameters()
        self.init_inner_iters()

    def init_inner_iters(self):
        '''calculate the number of inner iterations for each epoch'''
        self.trainset = build_dataset(self.hparams.data.train)
        self.inner_iters = len(self.trainset) // self.hparams.data.train_batch_size

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.valset = build_dataset(self.hparams.data.val)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.testset = build_dataset(self.hparams.data.test)

        if stage == 'predict' or stage is None:
            self.predictset = build_dataset(self.hparams.data.test)


    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.hparams.data.train_batch_size, sampler=self.trainset.data_sampler,
                          num_workers=self.hparams.data.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.hparams.data.val_batch_size, sampler=self.valset.data_sampler,
                          num_workers=self.hparams.data.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.hparams.data.test_batch_size, sampler=self.testset.data_sampler,
                          num_workers=self.hparams.data.num_workers, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.predictset, batch_size=self.hparams.data.test_batch_size, sampler=self.predictset.data_sampler,
                          num_workers=self.hparams.data.num_workers, shuffle=False)