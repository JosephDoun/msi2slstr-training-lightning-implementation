from torch.nn import AvgPool2d
from torch.nn import UpsamplingNearest2d
from torch.nn import Module
from torch import Tensor


StrictAvgDownSamplingModule = AvgPool2d(50, 50, 0, count_include_pad=False)
NonStrictAvgDownSamplingModule = AvgPool2d(80, 50, 15, count_include_pad=False)
UpsamplingModule = UpsamplingNearest2d(scale_factor=50)


class ValidAverageDownsampling(Module):
    """
    Creates a spatially (meaning in x,y axes) coarser copy of the provided
    array by averaging the valid (non zero) values.

    :param scale: The area of the spatial aggregation in number of elements
        (pixels.), defaults to 50.
    :type scale: int
    """

    def __init__(self, scale: int = 50) -> None:
        self.scale = int(scale)
    
    def _reshape(self, _tensor: Tensor):
        shape = _tensor.shape
        # Return shape: batch, channel, nytiles, nxtiles, tsize, tsize.
        return _tensor.reshape(shape[0], shape[1],
                               shape[2] // self.scale,
                               self.scale,
                               shape[3] // self.scale,
                               self.scale).swapaxes(-2, -3)

    def forward(self, _tensor: Tensor) -> Tensor:
        _tensor = self._reshape(_tensor)
        _sum = _tensor.sum((-1, -2))
        nzerocount = _tensor.gt(0).sum((-1, -2))
        return _sum.div(nzerocount.add(1e-10))


class SpatialAggregation(ValidAverageDownsampling):
    """
    Downsample an input image given a torch operation of choice.
    """
    def __init__(self, fn: str, scale: int = 50) -> None:
        super().__init__(scale)
        self.fn = fn

    def forward(self, _tensor: Tensor) -> Tensor:
        _tensor = self._reshape(_tensor)
        return getattr(_tensor, self.fn)((-1, -2))
