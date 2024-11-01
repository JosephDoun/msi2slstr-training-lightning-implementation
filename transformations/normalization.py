from torch import Tensor
from torch import tensor
from torch import float32
from torch import sqrt

from torch.nn import Module


class Normalizer(Module):
    """
    Module that scales the values of a given tensor according to the normalizer
    state, in a channelwise operation. i.e. Each channel (dim 1) of the
    array is scaled independently.

    .. math:: A_{\\mathrm{normal}} = \\frac{(A - offset)}
        {(scale + \\epsilon)}

    :param offset: A tuple of floats with the per channel value by which to
        offset the provided array's value range. Has to be broadcastable to
        the array's shape.
    :type offset: tuple[float]
    :param scale: A tuple of floats with the per channel value to scale the
        array's value range. Has to be broadcastable to the array's shape.
    :type scale: tuple[int]
    :param e: A small constant added to the denominator for numerical
        stability, defaults to 1e-5.
    :type e: float, optional

    .. automethod:: forward
    """

    def __init__(self, scale: tuple[float], offset: tuple[float] = (0.,), *,
                 e: float = 1e-5) -> None:
        super().__init__()
        self.offset = tensor(offset, dtype=float32).reshape(len(offset), 1, 1)
        self.scale = tensor(scale, dtype=float32).reshape(len(scale), 1, 1)
        self.e = e

    def forward(self, _t: Tensor) -> Tensor:
        """
        Execute normalization.

        :param array: Tensor to scale according to offset and scale.
        :param type: :class:`Tensor`

        :return: Normalized tesnor with scaled values.
        :rtype: :class:`Tensor`
        """
        return _t.add(self.offset).div(self.scale + self.e)

    def reverse(self, _t: Tensor) -> Tensor:
        """
        Inverse computation.

        :param array: Scaled tensor whose values to unscale.
        :param type: :class:`Tensor`

        :return: An inversely transformed tensor.
        :rtype: :class:`Tensor`
        """
        return _t.mul(self.scale + self.e).sub(self.offset)


def channel_stretch(_t: Tensor):
    xoffset = _t.sub(_t.amin((-1, -2), keepdim=True))
    return xoffset.div(xoffset.amax((-1, -2), keepdim=True) + 1e-6)


class Standardizer(Module):
    """
    Module that standardizes input to have mean 0 and variance 1, while
    preserving calculations to make an inverse computation possible.

    .. math:: ...

    .. automethod:: forward
    """

    def __init__(self,*, e: float = 1e-5) -> None:
        super().__init__()
        self.e = e

    def forward(self, _t: Tensor) -> Tensor:
        """
        Execute standardization.

        :param _t: Tensor to standardize.
        :param type: :class:`Tensor`

        :return: Standardized tensor.
        :rtype: :class:`Tensor`
        """
        self.mean = _t.mean((-1, -2), keepdim=True)
        self.var = _t.var((-1, -2), keepdim=True, unbiased=False)
        return _t.sub(self.mean).div(sqrt(self.var + self.e))

    def reverse(self, _t: Tensor,
                channels: int | tuple[int] | slice = slice(None)) -> Tensor:
        """
        Reverse the standardization of the last computation.

        :param _t: Lastly standardized tensor.
        :param type: :class:`Tensor`

        :return: Tensor with original values.
        :rtype: :class:`Tensor`
        """
        return _t.mul(sqrt(self.var[:, channels] + self.e))\
            .add(self.mean[:, channels])
