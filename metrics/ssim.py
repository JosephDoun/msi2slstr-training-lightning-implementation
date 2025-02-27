from torch.nn import Module
from torch import Tensor
from torch import stack

import torch


class ssim(Module):
    """
    Structural Similarity Index (basic implementation).
    
    *It is a design decision to keep component computations
    independent and allow handling by hardware cache.*

    Zhou Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli,
    "Image quality assessment: from error visibility to structural similarity,"
    in IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600-612,
    April 2004, doi: 10.1109/TIP.2003.819861.*

    :param dims: The dimensions to calculate the loss over.
    :type dims: tuple[int]
    """
    def __init__(self, dims: tuple = (-1, -2), *, agg: str = 'prod',
                 a: float = 1., b: float = 1., c: float = 1.) -> None:
        super().__init__()
        self._agg = agg
        self.dims = dims
        self.C = 1e-5
        self._a = a
        self._b = b
        self._c = c

    def _similarity(self, a: Tensor, b: Tensor) -> Tensor:
        return (
            # Numerator.
            a.mul(b).mul(2).add(self.C)
            # Denominator.
            .div(
                a.pow(2)
                .add(b.pow(2))
                .add(self.C)
                )
            .squeeze(self.dims))

    def l(self, x: Tensor, y: Tensor) -> Tensor:
        xmean = x.mean(self.dims, keepdim=True)
        ymean = y.mean(self.dims, keepdim=True)
        meanmin = xmean.detach().minimum(ymean.detach()).sub(.1)
        return self._similarity(xmean - meanmin,
                                ymean - meanmin).pow(self._a)

    def c(self, x: Tensor, y: Tensor) -> Tensor:
        return self._similarity(x.std(self.dims, keepdim=True),
                                y.std(self.dims, keepdim=True)).pow(self._b)

    def s(self, x: Tensor, y: Tensor) -> Tensor:
        xstd = x.std(self.dims, keepdim=True)
        ystd = y.std(self.dims, keepdim=True)
        return (
            # Numerator.
            # normalized x.
            x.sub(x.mean(self.dims, keepdim=True))
            .mul(# normalized y.
                 y.sub(y.mean(self.dims, keepdim=True)))
            .sum(self.dims, keepdim=True).div(
                Tensor([x.size(d) for d in self.dims]).prod() - 1
                )
            # Term necessary only when both tensors are 0.
            .add(self.C * .5 *

                 # Remove constant if any std non-zero.
                 ((~xstd.any(dim=self.dims, keepdim=True)) &
                  (~ystd.any(dim=self.dims, keepdim=True))))

            # Denominator.
            .div(
                    xstd.mul(ystd)

                    .add(self.C * .5)

                )).squeeze(self.dims).pow(self._c)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Averaged structural similarity loss with a range of values of
        [-1, 1] for maximization.
        """
        return stack([self.l(x, y), self.c(x, y), self.s(x, y)]).prod(0)


class ssim3d(ssim):
    """
    Includes the channel axis in the structural similarity measurement.
    """
    def __init__(self, *, agg: str = 'prod', a: float = 1,
                 b: float = 1, c: float = 1) -> None:
        super().__init__((-1, -2), agg=agg, a=a, b=b, c=c)
        self._channel_axis = ssim((-3,), agg=agg, a=a, b=b, c=c)
        self._aggregate = getattr(torch, agg)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        channel_axis = self._channel_axis(x, y).mean((-1, -2)).unsqueeze(-1)
        return stack([super().forward(x, y),
                      channel_axis.repeat(1, x.size(1))]).mean(0)
