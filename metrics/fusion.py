from torch import Tensor
from torch import stack
from torch import concat

from transformations.resampling import AvgDownSamplingModule
from transformations.resampling import UpsamplingModule

from .ssim import ssim

from config import DATA_CONFIG


class msi2slstr_loss(ssim):
    """
    A loss definition for the task of fusing S2 L1C & S3 RBT/LST images.

    Components: ...
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.signature = ssim((-3,))

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
        x = concat([x[:, DATA_CONFIG['sen2_bands']], thermal_estimate_x],
                   dim=-3)

        # Mask out irrelevant pixels using the green band.
        x = self._mask_x(x, y[:, [0]])

        # Not respecting zero pixels of y.
        # X corr minus Y residuals.
        structure = stack([self.s(x, Y_hat),
                           (1 - self.s(Y_hat, self._usample(y)).clamp(.1))])\
                            .mean(0)

        signature = self.signature(self._dsample(Y_hat), y)\
            .clamp(0)\
            .mean((-1, -2))\
            .unsqueeze(1)
        
        thermal = super().forward(thermal_estimate_y, y[:, 6:])
        
        thermal = thermal.mul(
            self.signature(thermal_estimate_y, y[:, 6:])
            .clamp(0)
            .mean((-1, -2))
            .unsqueeze(1)
            )

        # Respecting zero pixels of y but not of x. Not important.
        energy = super().forward(self._dsample(Y_hat), y)
        return energy.mul(signature).mul(structure), thermal

    def evaluate(self, y: Tensor, Y_hat: Tensor) -> Tensor:
        return super().evaluate(y, AvgDownSamplingModule(Y_hat))
    
    def __call__(self, x: Tensor, Y_hat: Tensor, y: Tensor,
                 thermal_estimate_y: Tensor,
                 thermal_estimate_x: Tensor) -> Tensor:
        return super().__call__(x, Y_hat, y, thermal_estimate_y,
                                thermal_estimate_x)
