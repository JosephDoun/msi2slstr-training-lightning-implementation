from lightning.pytorch.cli import LightningCLI
from datamodules.lightning import msi2slstr_datamodule # noqa: 
from modules.lightning import msi2slstr # noqa: 

import sys


def main():
    LightningCLI()
    return 0


if __name__ == "__main__":
    sys.exit(main())
