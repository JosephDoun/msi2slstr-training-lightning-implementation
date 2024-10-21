from lightning import LightningDataModule

from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch import Generator

from .dataset import msi2slstr_dataset


class msi2slstr_datamodule(LightningDataModule):
    """
    Data module for the msi2slstr project.
    """
    def prepare_data(self) -> None:
        super().prepare_data()

    def setup(self, stage: str) -> None:
        self.train, self.val, self.test = \
            random_split(msi2slstr_dataset(), [.8, .1, .1],
                         Generator().manual_seed(0))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test)
    
    def on_exception(self, exception: BaseException) -> None:
        return super().on_exception(exception)
    
    def teardown(self, stage: str) -> None:
        return super().teardown(stage)
