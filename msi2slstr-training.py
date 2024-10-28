from lightning.pytorch.cli import LightningCLI
from datamodules.lightning import msi2slstr_datamodule
from modules.lightning import msi2slstr

import sys


def main():
    LightningCLI(datamodule_class=msi2slstr_datamodule, trainer_defaults={})
    return 0


if __name__ == "__main__":
    sys.exit(main())
