from lightning.pytorch.cli import LightningCLI
# Register all experiments via dot imports.
import datamodules.fusion
import datamodules.emissivity
import modules.radiometry
import modules.emissivity

import sys

from torch import set_float32_matmul_precision

set_float32_matmul_precision('high')


def main():
    LightningCLI()
    return 0


if __name__ == "__main__":
    sys.exit(main())
