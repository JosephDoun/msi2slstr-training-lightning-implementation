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

        # Calculate scene maximum.
        scenemax = x.amax((-1, -2), keepdim=True)

        # Scale scene.
        composition = x.div(scenemax + 1e-5)

        # Calculate scene mean.
        scenemean = composition.mean((-1, -2), keepdim=True)

        # Offset scene to have mean 1.
        composition = composition - scenemean + 1

        # Map w/ adjusted mean and relative variance.
        fusion_map = composition * y.mean((-1, -2), keepdim=True)

        structure = self.s(Y_hat, fusion_map).mul(

            # Explicitly remove plane flatness.            
            (1 - self.s(Y_hat, self._usample(y)))

            )\
                .clamp(.0)

        thermal = super().forward(thermal_estimate_y, y[:, 6:])

        thermal = thermal.add(
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

        return energy.mul(signature).mul(structure), thermal
    
    def __call__(self, x: Tensor, Y_hat: Tensor, y: Tensor,
                 thermal_estimate_y: Tensor,
                 thermal_estimate_x: Tensor) -> Tensor:
        return super().__call__(x, Y_hat, y, thermal_estimate_y,
                                thermal_estimate_x)
