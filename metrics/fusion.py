from torch import Tensor
from torch import concat

from transformations.resampling import DSamplingModule

from .ssim import ssim


class msi2slstr_loss(ssim):
    """
    A loss definition for the task of fusing S2 L1C & S3 RBT/LST images.

    Components: ...
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @staticmethod
    def _dsample(x: Tensor):
        return DSamplingModule(x)

    def forward(self, x: Tensor, Y_hat: Tensor, y: Tensor,
                thermal_estimate_y: Tensor, thermal_estimate_x: Tensor):
        """
        :returns: Collective loss, energy loss and thermal loss in a 3-tuple.
        :rtype: tuple
        """
        x = concat([x[:, [2, 3, 8, 10, 11, 12]], thermal_estimate_x], dim=-3)
        structure = self.s(x, Y_hat)
        # Scalar.
        thermal = super().forward(thermal_estimate_y, y[:, 6:]).mean()
        energy = super().forward(y, self._dsample(Y_hat))
        return sum([energy, structure, thermal]).div(3.)

    def energy(self, x: Tensor, y: Tensor) -> Tensor:
        return super().evaluate(x, DSamplingModule(y))
