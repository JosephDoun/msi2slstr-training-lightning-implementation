from torch import Tensor
from typing import Any


class Average(Tensor):
    """
    Tensor subclass with overriden add operator and internal counter
    that keeps track of a running mean.
    """

    def __init__(self, *args, **kwargs):
        self.count = 1

    def __iadd__(self, other: Any) -> Tensor:
        self.count += 1
        return self.add_((other - self) / self.count)

    def __add__(self, other: Any) -> Tensor:
        count = self.count + 1
        other = self.add((other - self) / count)
        other.count = count
        return other

    def __mul__(self, other: Any) -> Tensor:
        raise NotImplementedError()

    def __imul__(self, other: Any) -> Tensor:
        raise NotImplementedError()

    def __rmul__(self, other: Any) -> Tensor:
        raise NotImplementedError()

    def __div__(self, other: Any) -> Tensor:
        raise NotImplementedError()

    def __idiv__(self, other: Any) -> Tensor:
        raise NotImplementedError()

    def __rdiv__(self, other):
        raise NotImplementedError()

    def __matmul__(self, other: Any) -> Tensor:
        raise NotImplementedError()
