from torch import Tensor
from torch import concat

from transformations.resampling import AvgDownSamplingModule
from transformations.resampling import UpsamplingModule

from .ssim import ssim


class msi2slstr_loss(ssim):
    """
    A loss definition for the task of fusing S2 L1C & S3 RBT/LST images.

    Components: ...
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _dsample(self, x: Tensor):
        return AvgDownSamplingModule(x)
    
    def _usample(self, x: Tensor):
        return UpsamplingModule(x)
    
    def _mask_x(self, high_x: Tensor, low_y: Tensor):
        return high_x.mul(UpsamplingModule(low_y).gt(0))

    def forward(self, x: Tensor, Y_hat: Tensor, y: Tensor,
                thermal_estimate_y: Tensor, thermal_estimate_x: Tensor):
        """
        :returns: Collective loss, energy loss and thermal loss in a 3-tuple.
        :rtype: tuple
        """
        # zero y pixels must be masked out of x for consistency.
        x = concat([x[:, [2, 3, 8, 10, 11, 12]], thermal_estimate_x], dim=-3)
        # Mask out irrelevant pixels using LST band.
        x = self._mask_x(x, y[:, [-1]])
        # Not respecting zero pixels pf y.
        structure = self.s(x, Y_hat)
        # Scalar.
        thermal = super().forward(thermal_estimate_y, y[:, 6:]).mean()
        # Respecting zero pixels of y but not of x. Not important.
        energy = super().forward(y, self._dsample(Y_hat))
        return sum([energy, structure, thermal]).div(3.)

    def energy(self, x: Tensor, y: Tensor) -> Tensor:
        return super().evaluate(x, AvgDownSamplingModule(y))
