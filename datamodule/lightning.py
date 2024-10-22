from lightning import LightningDataModule

from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch import Generator

from .dataset import msi2slstr_dataset


class msi2slstr_datamodule(LightningDataModule):
    """
    Data module for the msi2slstr project.
    """
    def __init__(self, batch_size: int = 32) -> None:
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        super().prepare_data()

    def setup(self, stage: str) -> None:
        self.train, self.val, self.test = \
            random_split(msi2slstr_dataset(), [.8, .1, .1],
                         Generator().manual_seed(0))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, batch_size=self.batch_size,
                          pin_memory=True, num_workers=2)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val, batch_size=8, num_workers=2)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test, batch_size=8)
    
    def on_exception(self, exception: BaseException) -> None:
        return super().on_exception(exception)
    
    def teardown(self, stage: str) -> None:
        return super().teardown(stage)
