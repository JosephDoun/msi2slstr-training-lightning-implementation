"""
Experiment 2 datamodule definitions.
"""

from lightning import LightningDataModule
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS

from torch.utils.data import random_split
from torch.utils.data import DataLoader

from torch import Generator

from .datasets import msi2slstr_dataset
from .datasets import independent_pairs
from .datasets import predictor_dataset


class autoencoding_datamodule(LightningDataModule):
    """
    ...
    """
    def __init__(self, data="data", batch_size=32) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["_class_path"])

    def setup(self, stage: str) -> None:
        self.train, self.valid = random_split(independent_pairs(), [.9, .1],
                                              Generator().manual_seed(0))
        self.test = msi2slstr_dataset()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, 32, shuffle=True, pin_memory=True,
                          num_workers=4)
    
    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.valid, 32, pin_memory=True, num_workers=4)
    
    def test_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.test, 32, pin_memory=True, num_workers=4)
    
    def predict_dataloader(self) -> DataLoader:
        return DataLoader(predictor_dataset(self.hparams.datadir),
                          batch_size=1,
                          pin_memory=True,
                          num_workers=2)
    
    def on_exception(self, exception: BaseException) -> None:
        return super().on_exception(exception)
    
    def teardown(self, stage: str) -> None:
        return super().teardown(stage)
