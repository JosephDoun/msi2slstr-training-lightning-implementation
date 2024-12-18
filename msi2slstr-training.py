from lightning.pytorch.cli import LightningCLI
# Register all experiments via dot imports.
import datamodules.experiment_1
import datamodules.experiment_2
import datamodules.emissivity
import modules.experiment_1
import modules.experiment_2
import modules.emissivity

import sys

from torch import set_float32_matmul_precision

set_float32_matmul_precision('high')


def main():
    LightningCLI()
    return 0


if __name__ == "__main__":
    sys.exit(main())
