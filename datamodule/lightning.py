from lightning import LightningDataModule

from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch import Generator

from .dataset import msi2slstr_dataset


class msi2slstr_datamodule(LightningDataModule):
    """
    Data module for the msi2slstr project.
    """
    def __init__(self, batch_size: int = 32, datadir: str = 'data',
                 num_workers: int = 1) -> None:
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self) -> None:
        super().prepare_data()

    def setup(self, stage: str) -> None:
        self.train, self.val, self.test = \
            random_split(msi2slstr_dataset(self.hparams.datadir),
                         [.8, .1, .1],
                         Generator().manual_seed(0))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, batch_size=self.hparams.batch_size,
                          pin_memory=True,
                          shuffle=True,
                          num_workers=self.hparams.num_workers)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val,
                          # Each batch requires a system call because
                          # gdal is not thread-safe. A large batch_size
                          # creates a bottle neck. NOTE.
                          batch_size=4,
                          pin_memory=True,
                          num_workers=4)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test, batch_size=4)
    
    def on_exception(self, exception: BaseException) -> None:
        return super().on_exception(exception)
    
    def teardown(self, stage: str) -> None:
        return super().teardown(stage)
