from torch import Tensor


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
        self.offset = Tensor(offset).reshape(1, len(offset), 1, 1)
        self.scale = Tensor(scale).reshape(1, len(scale), 1, 1)
        self.e = e

    def __call__(self, array: Tensor) -> Tensor:
        """
        Execute normalization.

        :param array: Array to rescale according to offset and scale.
        :param type: :class:`Tensor`

        :return: Normalized array with rescaled values.
        :rtype: :class:`Tensor`
        """
        return (array - self.offset) / (self.scale + self.e)

    def reverse(self, array: Tensor) -> Tensor:
        """
        Reverse the value normalization.

        :param array: Scaled array whose values to unscale.
        :param type: :class:`Tensor`

        :return: Array with original values.
        :rtype: :class:`Tensor`
        """
        return array * (self.scale + self.e) + self.offset
