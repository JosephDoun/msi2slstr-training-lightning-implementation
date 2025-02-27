import torch.nn as nn

from torch import einsum
from torch import concat
from torch import Tensor
from torch import arange
from torch import randperm
from torch import ones
from torch import zeros
from torch import randn_like

from torch.nn import Parameter
from torch.nn import PReLU as Activation

from config import MODEL_CONFIG as CONFIG
from config import DATA_CONFIG as DATA

from typing import Any, Tuple

from torch.nn.functional import batch_norm
from torch.nn.functional import instance_norm
from torch.nn.functional import avg_pool3d


class StdConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int | Tuple[int, int],
                 stride: int | Tuple[int, int] = 1,
                 padding: str | int | Tuple[int, int] = 0,
                 dilation: int | Tuple[int, int] = 1, groups: int = 1,
                 bias: bool = True, padding_mode: str = 'zeros',
                 device=None, dtype=None) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode,
                         device, dtype)

    def forward(self, x: Tensor):
        weight = instance_norm(self.weight[None, ...],
                               None,            # Running mean.
                               None,            # Running var.
                               None,            # Weight.
                               None,            # Bias.
                               True,            # Use input stats.
                               0,               # Momentum.
                               1e-8)[0]

        return self._conv_forward(x, weight, self.bias)


class StaticNorm2D(nn.Module):
    """
    Statically scales input data according to internal state values.

    :param key: Dictionary key for stats loading, either "sen3" or "sen2".
    :type key: str
    """
    def __init__(self, key: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        mean = Tensor(DATA['stats'][key]['mean'])\
            .reshape(1, -1, 1, 1)
        var = Tensor(DATA['stats'][key]['var'])\
            .reshape(1, -1, 1, 1)
        self.e = 1e-5
        self.register_buffer("mean", mean)
        self.register_buffer("var", var)

    def forward(self, x: Tensor):
        return (x.sub(self.mean)
            .div(self.var + self.e))

    def denorm(self, x: Tensor, channels: slice | list[int] = slice(None)):
        return x.mul(self.var[:, channels].add(self.e))\
            .add(self.mean[:, channels])


class BGNorm(nn.Module):
    def __init__(self, _in: int, groups: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert not _in % groups, "in channels not divisible by groups."
        self._groups = groups
        self.register_buffer("mu", zeros(groups, requires_grad=False))
        self.register_buffer("sigma", ones(groups, requires_grad=False))
        self.register_parameter("a", Parameter(ones(groups)))
        self.register_parameter("b", Parameter(zeros(groups)))

    def forward(self, x: Tensor):
        shape = x.shape
        x = x.view(shape[0], self._groups, -1)
        return batch_norm(x, # This shape should be (B, G, -1)
                          self.mu,
                          self.sigma,
                          self.a,
                          self.b,
                          self.training,
                          0.01,
                          1e-8,).view(shape)


class ReflectiveToEmissive(nn.Module):
    """
    Given the reflective part of the spectrum corresponding to measurements
    of the first 6 bands of a Sentinel-3 RBT product, estimate the thermal
    part of the spectrum.

    NOTE: This module should be refactored to take advantage of the additional
    radiometric information contained in Sentinel-2 MSI measurements.
    """
    def __init__(self, _in: int, _out: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.module = nn.Sequential(
            nn.Conv2d(_in, 16, 1),
            ResBlock(16, 32, kernel_size=1, padding=0, groups=8),
            nn.BatchNorm2d(32),
            Activation(),
            nn.Conv2d(32, _out, 1),
            nn.BatchNorm2d(_out)
        )

    def forward(self, x: Tensor):
        return self.module(x)


class SNE(nn.Module):
    def __init__(self, _in, fn: str = 'mean', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.module = nn.Sequential(
            nn.Linear(_in, _in // 8, bias=False),
            Activation(),
            nn.Linear(_in // 8, _in, bias=False),
            nn.Sigmoid(),
        )
        self._fn = fn

    def forward(self, x: Tensor):
        return x.mul(self.module(getattr(x, self._fn)((-1, -2)))
                     .view(x.size(0), x.size(1), 1, 1))


class ReScale2D(nn.Module):
    def __init__(self, _in, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.norm = nn.GroupNorm(_in, _in, affine=False)

    def forward(self, x: Tensor, ref: Tensor):
        """
        Linearly project `x` to the radiometric scale of `y`;
        """
        return (self.norm(x)
                .mul(ref.std((-1, -2), keepdim=True))
                .add(ref.mean((-1, -2), keepdim=True)))

    def __call__(self, x: Tensor, ref: Tensor, *args: Any, **kwds: Any) -> Any:
        return super().__call__(x, ref, *args, **kwds)


class UpsamplingBlock(nn.Module):
    def __init__(self, _in: int, _out: int, size: int,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.up = nn.UpsamplingNearest2d((size, size))
        self.attend = LocalAttention(_in, _out // 2)
        self.module = ResBlock(_in + _out // 2, _out, stride=1)

    def forward(self, x, connection):
        xc = self.attend(x, connection)
        return self.module(concat([self.up(xc),
                                   connection], dim=-3))


class DownsamplingBlock(nn.Module):
    """
    Residual downsampling block.
    """
    def __init__(self, _in: int, _out: int, groups: int = 1,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.module = ResBlock(_in, _out, stride=2, padding=1, groups=groups)

    def forward(self, x):
        return self.module(x)


class Head(nn.Module):
    def __init__(self, _in: int, _out: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.module = nn.Sequential(
            ASPP(_in, _in // 2, 1, 6, 12),
            nn.BatchNorm2d(_in // 2),
            Activation(),
            nn.Conv2d(_in // 2, _out, 1),
            nn.BatchNorm2d(_out),
        )

    def forward(self, x):
        return self.module(x)


class Bridge(nn.Module):
    def __init__(self, _in: int, _out: int, size: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.module = ASPP(_in, _out, 1, 3, 6)
        self.y_expansion = ChannelExpansion(size, 12, _in)

    def forward(self, x: Tensor, y: Tensor):
        return self.module(x + self.y_expansion(y))


class Stem(nn.Module):
    def __init__(self, _in: list[int, int], _out: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.module = nn.Sequential(
            nn.Conv2d(_in[0], _out, kernel_size=3, padding=1,
                      padding_mode="reflect"),
            nn.BatchNorm2d(_out),
            Activation(),
            nn.Conv2d(_out, _out, kernel_size=3, padding=1,
                      padding_mode="reflect"),
        )

        self._mixture = nn.Sequential(
            nn.Conv2d(_in[1], _out, 3, padding=1,
                      padding_mode='reflect'),
            nn.BatchNorm2d(_out),
        )

    def forward(self, x: list[Tensor, Tensor]):
        return self.module(x[0]).add(self._mixture(x[1]))


class ChannelExpansion(nn.Module):
    def __init__(self, size: int, _in: int, _out: int) -> None:
        super().__init__()
        assert not max(_in, _out) % min(_in, _out), "Not divisible."
        self.resample = nn.UpsamplingNearest2d((size, size))
        self._range = _out // _in

    def forward(self, x: Tensor):
        x = self.resample(x)
        # Repeat channels _in // _out times per channel.
        idx = arange(x.size(-3)).reshape(-1, 1)\
            .repeat(1, self._range).flatten()
        return x[:, idx]


class ChannelCollapse(nn.Module):
    def __init__(self, size: int, _in: int, _out: int) -> None:
        super().__init__()
        assert not max(_in, _out) % min(_in, _out), "Not divisible."
        self.resample = nn.UpsamplingBilinear2d((size, size))
        self._out = _out
        self._groups = _in // _out

    def forward(self, x: Tensor):
        x = self.resample(x)
        return x.view(x.size(0), self._out, self._groups,
                      x.size(2), x.size(3)).mean(2)


class LocalAttention(nn.Module):
    def __init__(self, _in: int, conn_in: int,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.query_x = nn.Sequential(
            nn.BatchNorm2d(_in),
            Activation(),
            nn.Conv2d(_in, _in,
                      kernel_size=3,
                      padding=1,
                      padding_mode="reflect"))

        # Stride 2
        self.query_c = nn.Sequential(
            nn.BatchNorm2d(conn_in),
            Activation(),
            nn.Conv2d(conn_in, _in,
                      kernel_size=3,
                      padding=1,
                      padding_mode="reflect",
                      stride=2))

        # SoftMax?
        self.sum = nn.Sequential(
            nn.BatchNorm2d(_in),
            Activation(),
            nn.Conv2d(_in, _in,
                      kernel_size=3,
                      padding=1,
                      padding_mode="reflect"),
            nn.Sigmoid())

    def forward(self, x, connection):
        query_x = self.query_x(x)
        query_c = self.query_c(connection)
        multiplier = self.sum(query_c.add(query_x))
        return x.mul(multiplier)


class ASPP(nn.Module):
    def __init__(self, _in: int, _out: int, scale_1: int, scale_2: int,
                 scale_3: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scale_1 = nn.Sequential(
            nn.Conv2d(_in, _out, kernel_size=3, dilation=scale_1,
                      padding=scale_1, padding_mode="reflect"),
            nn.BatchNorm2d(_out),
        )

        self.scale_2 = nn.Sequential(
            nn.Conv2d(_in, _out, kernel_size=3, dilation=scale_2,
                      padding=scale_2, padding_mode="reflect"),
            nn.BatchNorm2d(_out),
        )

        self.scale_3 = nn.Sequential(
            nn.Conv2d(_in, _out, kernel_size=3, dilation=scale_3,
                      padding=scale_3, padding_mode="reflect"),
            nn.BatchNorm2d(_out),
        )

        self.pool = nn.Conv2d(_out, _out, kernel_size=1)

    def forward(self, x):
        x1 = self.scale_1(x)
        x2 = self.scale_2(x)
        x3 = self.scale_3(x)
        return self.pool(x1 + x2 + x3)


class ResBlock(nn.Module):
    def __init__(self, _in: int, _out: int, *, kernel_size: int = 3,
                 stride: int = 1, dilation: int = 1, padding: int = 1,
                 groups: int = 1, **kwargs: dict) -> None:
        super().__init__()
        self.module = nn.Sequential(
            nn.BatchNorm2d(_in),
            Activation(),
            nn.Conv2d(in_channels=_in, out_channels=_out,
                      kernel_size=kernel_size,
                      padding=padding,
                      stride=stride,
                      dilation=dilation,
                      padding_mode="reflect"),
            nn.BatchNorm2d(_out),
            Activation(),
            nn.Conv2d(in_channels=_out, out_channels=_out,
                      kernel_size=kernel_size,
                      padding=1 if kernel_size == 3 else padding,
                      padding_mode="reflect"))

        self.residual = nn.Sequential(
            nn.Conv2d(_in, _out, 1, stride=stride, groups=16),
            nn.BatchNorm2d(_out),
        )

    def forward(self, x):
        return self.module(x).add(self.residual(x))
