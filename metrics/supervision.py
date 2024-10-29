
from torch.nn import Module
from torch.nn import AvgPool3d
from torch.nn import ModuleList

from torch import Tensor



class DeepLoss(Module):
    """
    Returns the accumulated loss of a deep supervision task.
    """
    def __init__(self, in_features: list[int], sizes: list[int],
                 loss_fn: callable, maximize: bool = False,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        #: List of resizing modules to y dimensions.
        self.modulelist = ModuleList([
            # y dims 12, 2, 2.
                AvgPool3d((_in // 12, size // 2, size // 2),
                          (_in // 12, size // 2, size // 2),
                          count_include_pad=False)
                for _in, size in zip(in_features, sizes)
            ])

        self._fn = loss_fn

    def forward(self, *x: Tensor, y: Tensor):

        y = y.view(y.size(0), y.size(1), -1)
        loss = 0

        for module, x_tensor in zip(self.modulelist, x, strict=True):
            x_tensor = module(x_tensor)
            x_tensor = x_tensor.view(x_tensor.size(0), x_tensor.size(1), -1)
            loss += self._fn(x_tensor, y).mean(-1)

        return loss
