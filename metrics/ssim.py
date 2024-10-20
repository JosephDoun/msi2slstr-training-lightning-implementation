from torch.nn import Module
from torch import Tensor


class ssim(Module):
    """
    Structural Similarity Index (basic implementation).
    
    *It is a design decision to keep component computations
    independent and allow handling by hardware cache.*

    :param dims: The dimensions to calculate the loss over.
    :type dims: tuple[int]
    """
    def __init__(self, dims: tuple=(-1, -2), *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dims = dims
        self.C = 1e-10

    def _similarity(self, a: Tensor, b: Tensor) -> Tensor:
        return (
            # Numerator.
            a.mul(b).mul(2).add(self.C)
            # Denominator.
            .div(a.pow(2).add(b.pow(2)).add(self.C))
            ).squeeze(self.dims)

    def l(self, x: Tensor, y: Tensor) -> Tensor:
        return self._similarity(x.mean(self.dims, keepdim=True),
                                y.mean(self.dims, keepdim=True))
    
    def c(self, x: Tensor, y: Tensor) -> Tensor:
        return self._similarity(x.std(self.dims, keepdim=True),
                                y.std(self.dims, keepdim=True))
    
    def s(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Biased implementation for performance.
        """
        return (
            # Numerator.
            # normalized x.
            x.sub(x.mean(self.dims, keepdim=True))
            .mul(# normalized y.
                y.sub(y.mean(self.dims, keepdim=True)))
            .mean(self.dims, keepdim=True)
            # .add(self.C)
            # Denominator.
            .div(
                    x.std(self.dims, keepdim=True)
                    .mul(y.std(self.dims, keepdim=True))
                    
                    .add(self.C)

                )).squeeze(self.dims)
    
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Compute an unrestricted version of SSIM that
        takes values between -1 and 3.
        """
        return sum([self.l(x, y), self.c(x, y), self.s(x, y)])
