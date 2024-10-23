from torch import Tensor
from torch import tensor
from torch import float32


class Normalizer:
    """
    Class to normalize given array according to the normalizer state, always in
    a channelwise manner. i.e. Each channel (dim 1) of the array is normalized
    independently.

    .. math:: A_{\\mathrm{normal}} = \\frac{(A - offset)}
        {(scale + \\varepsilon)}

    :param offset: A tuple of floats with the per channel value by which to
        offset the provided array's value range. Has to be broadcastable to
        the array's shape.
    :type offset: tuple[float]
    :param scale: A tuple of floats with the per channel value to scale the
        array's value range. Has to be broadcastable to the array's shape.
    :type scale: tuple[int]
    :param e: A small constant added to the denominator to avoid division by 0,
        defaults to 1e-15.
    :type e: float, optional

    .. automethod:: __call__
    """

    def __init__(self, scale: tuple[float], offset: tuple[float], *,
                 e: float = 1e-15) -> None:
        self.offset = tensor(offset, dtype=float32).reshape(len(offset), 1, 1)
        self.scale = tensor(scale, dtype=float32).reshape(len(scale), 1, 1)
        self.e = e

    def __call__(self, array: Tensor) -> Tensor:
        """
        Execute normalization.

        :param array: Array to rescale according to offset and scale.
        :param type: :class:`Tensor`

        :return: Normalized array with rescaled values.
        :rtype: :class:`Tensor`
        """
        return array.add(self.offset).div(self.scale + self.e)

    def reverse(self, array: Tensor) -> Tensor:
        """
        Reverse the value normalization.

        :param array: Scaled array whose values to unscale.
        :param type: :class:`Tensor`

        :return: Array with original values.
        :rtype: :class:`Tensor`
        """
        return array.mul(self.scale + self.e).add(self.offset)


def channel_stretch(x: Tensor):
    return x.sub(x.amin((-1, -2))).div(x.amax((-1, -2)) + 1e-10)
