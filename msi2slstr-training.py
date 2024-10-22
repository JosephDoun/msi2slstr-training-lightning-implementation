from lightning.pytorch.cli import LightningCLI
from datamodule.lightning import msi2slstr_datamodule
from modules.lightning import msi2slstr

import sys


def main():
    LightningCLI(msi2slstr, msi2slstr_datamodule)
    return 0


if __name__ == "__main__":
    sys.exit(main())
