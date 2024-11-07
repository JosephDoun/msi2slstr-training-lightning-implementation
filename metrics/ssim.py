from torch.nn import Module
from torch import Tensor
from torch import stack

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
    def __init__(self, dims: tuple = (-1, -2), *,
                 a: float = 1., b: float = 1., c: float = 1.) -> None:
        super().__init__()
        self.dims = dims
        self.C = 1e-3
        self._a = a
        self._b = b
        self._c = c

    def _similarity(self, a: Tensor, b: Tensor) -> Tensor:
        return (
            # Numerator.
            a.mul(b).mul(2).add(self.C)
            # Denominator.
            .div(a.pow(2).add(b.pow(2)).add(self.C))
            .squeeze(self.dims))

    def l(self, x: Tensor, y: Tensor) -> Tensor:
        return self._similarity(x.mean(self.dims, keepdim=True),
                                y.mean(self.dims, keepdim=True)).clamp(0)\
                                ** self._a
    
    def c(self, x: Tensor, y: Tensor) -> Tensor:
        return self._similarity(x.std(self.dims, keepdim=True),
                                y.std(self.dims, keepdim=True)) ** self._b
    
    def s(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Biased implementation for performance.
        """
        xnorm = x.sub(x.mean(self.dims, keepdim=True))
        ynorm = y.sub(y.mean(self.dims, keepdim=True))
        return (
            # Numerator.
            # normalized x.
            xnorm
            .mul(  # normalized y.
                   ynorm)
            .sum(self.dims, keepdim=True).div(x.size(-1) * x.size(-2))
            # Term necessary only when both tensors are 0.
            .add(self.C * .5 * ((~xnorm.any(dim=self.dims, keepdim=True)) &
                                (~ynorm.any(dim=self.dims, keepdim=True))))
            # Denominator.
            .div(
                    x.std(self.dims, keepdim=True)
                    .mul(y.std(self.dims, keepdim=True))
                    
                    .add(self.C * .5)

                )).squeeze(self.dims) ** self._c
    
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Averaged structural similarity loss with a range of values of
        [-1, 1] for maximization.
        """
        return stack([self.l(x, y), self.c(x, y), self.s(x, y)]).mean(0)
    
    def evaluate(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Multiplied criterion for verification.
        """
        x = x.detach()
        y = y.detach()
        return stack([self.l(x, y), self.c(x, y), self.s(x, y)]).prod(0)
