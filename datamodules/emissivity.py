from .experiment_1 import msi2slstr_datamodule
from .experiment_1 import random_split
from .datasets import msi2slstr_dataset

from torch import Generator


class emissivity_datamodule(msi2slstr_datamodule):
    """
    Use Sentinel-3 images in combination with corregistered Sentinel-2
    acquisitions for estimating energy emissions measured by the SLSTR sensor.
    """
    def __init__(self, batch_size: int = 32, datadir: str = 'data',
                 num_workers: int = 4) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['_class_path'])

    def setup(self, stage: str) -> None:
        self.train, self.val = \
            random_split(msi2slstr_dataset(dirname='data', t_size=(100, 2)),
                         [.9, .1],
                         Generator().manual_seed(0))