import sys
sys.path.append(".")
    
import torch

from modules.lightning import msi2slstr


def export(args: list[str]):
    model = [path for path in args if path.endswith(".ckpt")]

    m = msi2slstr.load_from_checkpoint(model[0] if model else
                                       "models/last.ckpt")

    m.to_onnx("exports/msi2slstr.onnx",
              input_sample=(torch.randn(32, 13, 100, 100),
                            torch.randn(32, 12, 2, 2)))

    return 0


if __name__ == "__main__":
    sys.exit(export(sys.argv))
