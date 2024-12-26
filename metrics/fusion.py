from torch import Tensor
from torch import concat

from transformations.resampling import StrictAvgDownSamplingModule
from transformations.resampling import UpsamplingModule

from .ssim import ssim

from config import DATA_CONFIG


class fusion_energy_metric(ssim):
    """
    Validation metric. Evaluate degree of energy preservation in the
    fused image.
    """
    def forward(self, y: Tensor, Y_hat: Tensor):
        return super().forward(y, StrictAvgDownSamplingModule(Y_hat))


class fusion_topo_metric(ssim):
    """
    Validation metric. Evaluate the degree of the preserved topological
    information in the fused image.
    """
    def forward(self, x: Tensor, Y_hat: Tensor) -> Tensor:
        return super().s(x, Y_hat)

