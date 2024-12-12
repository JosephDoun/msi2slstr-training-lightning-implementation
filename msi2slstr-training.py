from lightning.pytorch.cli import LightningCLI
# Register all experiments via dot imports.
import datamodules.experiment_1
import datamodules.experiment_2
import modules.experiment_1
import modules.experiment_2

import sys


def main():
    LightningCLI()
    return 0


if __name__ == "__main__":
    sys.exit(main())
