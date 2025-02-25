"""
Dataset module definitions for experiment concept 1.
"""
from lightning import LightningDataModule

from torch.utils.data import random_split
from torch.utils.data import DataLoader

from torch import Generator

from .datasets import msi2slstr_dataset
from .datasets import sen3dataset
from .datasets import predictor_dataset


class msi2slstr_datamodule(LightningDataModule):
    def __init__(self, batch_size: int, 
                 t_size: tuple[int, int], datadir: str = 'data',
                 num_workers: int = 4) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["_class_path"])

    def prepare_data(self) -> None:
        super().prepare_data()

    def setup(self, stage: str) -> None:
        self.train, self.val, self.test = \
            random_split(msi2slstr_dataset(self.hparams.datadir),
                         [.9, .05, .05],
                         Generator().manual_seed(0))

    def train_dataloader(self) -> DataLoader:
        # TODO add sampler
        return DataLoader(self.train, batch_size=self.hparams.batch_size,
                          pin_memory=True,
                          shuffle=True,
                          num_workers=self.hparams.num_workers,
                          sampler=None,
                          batch_sampler=None,
                          prefetch_factor=2)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True,
                          num_workers=15,
                          prefetch_factor=4)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True,
                          prefetch_factor=2)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(predictor_dataset(self.hparams.datadir,
                                            self.hparams.t_size),
                          batch_size=self.hparams.batch_size,
                          pin_memory=True,
                          num_workers=self.hparams.num_workers,
                          prefetch_factor=2)

    def on_exception(self, exception: BaseException) -> None:
        return super().on_exception(exception)

    def teardown(self, stage: str) -> None:
        return super().teardown(stage)
