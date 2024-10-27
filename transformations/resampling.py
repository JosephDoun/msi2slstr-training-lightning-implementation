from torch.nn import AvgPool2d
from torch.nn import UpsamplingNearest2d
from torch.nn import Module
from torch import Tensor


AvgDownSamplingModule = AvgPool2d(50, 50, 0)
UpsamplingModule = UpsamplingNearest2d(scale_factor=50)


class ValidAverageDownsampling(Module):
    """
    Creates a spatially (meaning in x,y axes) coarser copy of the provided
    array by averaging the valid values.

    :param scale: The area of the spatial aggregation in number of elements
        (pixels.), defaults to 50.
    :param scale: int
    """

    def __init__(self, scale: int = 50) -> None:
        self.scale = int(scale)

    def forward(self, _tensor: Tensor) -> Tensor:
        shape = _tensor.shape
        _tensor = _tensor.reshape(shape[0], shape[1],
                              shape[2] // self.scale,
                              self.scale,
                              shape[3] // self.scale,
                              self.scale).swapaxes(-2, -3)
        _sum = _tensor.sum((-1, -2))
        nzerocount = _tensor.gt(0).sum((-1, -2))
        return _sum.div(nzerocount.add(1e-10))
