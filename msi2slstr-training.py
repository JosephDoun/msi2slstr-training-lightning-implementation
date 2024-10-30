from lightning.pytorch.cli import LightningCLI
from datamodules.lightning import msi2slstr_datamodule # noqa: F401
from modules.lightning import msi2slstr # noqa: F401

import sys


def main():
    LightningCLI()
    return 0


if __name__ == "__main__":
    sys.exit(main())
