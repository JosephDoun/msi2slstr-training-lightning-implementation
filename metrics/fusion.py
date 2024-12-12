from torch import Tensor
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
        #: Channel axis ssim to evaluate
        #  pixel spectral signatures.
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

        # Scale scene to max 1.
        topology = x.div(x.amax((-1, -2), keepdim=True) + 1e-5)

        # Offset scene to have mean 1.
        topology = topology - topology.mean((-1, -2), keepdim=True) +1

        # Map w/ adjusted mean and accordingly scaled variance.
        # Scale each channel to have the mean of y.
        # Can also try pixel (y) based mean adjustment. TODO.
        topology = topology * y.mean((-1, -2), keepdim=True)

        # Balance between high-res correlation
        # and low-res decorrelation conditions.
        topology = self.s(Y_hat, topology).mul(

            # Explicitly resist plane flatness.
            # (Maintain subpixel variance as high
            #  as permitted by high-res correlation.)
            (1 - self.s(Y_hat, self._usample(y))) ** 2

            )\
                .mul(self.c(Y_hat, topology) ** .2)\
                .clamp(.0)

        thermal = super().forward(thermal_estimate_y, y[:, 6:])

        thermal = thermal.add(
            # Include channel-axis pixel signature similarity.
            self.signature(thermal_estimate_y, y[:, 6:])
            .clamp(0)
            .mean((-1, -2))
            .unsqueeze(1)
            )

        # Respecting zero pixels of y but not of x. Not important.
        energy = super().forward(self._dsample(Y_hat), y)
        signature = self.signature(self._dsample(Y_hat), y)\
            .clamp(0)\
            .mean((-1, -2))\
            .unsqueeze(1)\

        return energy.mul(signature).mul(topology), thermal

    def __call__(self, x: Tensor, Y_hat: Tensor, y: Tensor,
                 thermal_estimate_y: Tensor,
                 thermal_estimate_x: Tensor) -> Tensor:
        return super().__call__(x, Y_hat, y, thermal_estimate_y,
                                thermal_estimate_x)
