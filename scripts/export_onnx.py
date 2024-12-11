import sys
sys.path.append(".")
    
import torch

from modules.experiment_1 import msi2slstr_predict


def export(args: list[str]):
    ckpt = [arg for arg in args if arg.endswith(".ckpt")]
    ckpt = ckpt[0] if ckpt else None

    model = msi2slstr_predict()
    model._load_from_state_dict(
        torch.load(ckpt or "models/last.ckpt", weights_only=True),
        "", {}, True, None, [], []
    )

    model.to_onnx("exports/msi2slstr.onnx",
                  input_sample=(torch.randn(1, 13, 500, 500),
                                torch.randn(1, 12,  10,  10)))

    return 0


if __name__ == "__main__":
    sys.exit(export(sys.argv))
