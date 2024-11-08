import torch.nn as nn

from torch import einsum
from torch import concat
from torch import Tensor

from torch.nn import LeakyReLU as Activation

from config import MODEL_CONFIG as CONFIG
from config import DATA_CONFIG as DATA

from typing import Any


class StaticNorm2D(nn.Module):
    """
    Class providing the denorm method, reversing BatchNorm2D.

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
        return x.sub(self.mean)\
            .div(self.var + self.e)

    def denorm(self, x: Tensor, channels: slice | list[int] = slice(None)):
        return x.mul(self.var[:, channels] + self.e)\
            .add(self.mean[:, channels])


class OpticalToThermal(nn.Module):
    def __init__(self, _in: int, _out: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = nn.Sequential(
            nn.Conv2d(_in, _in * 2, 1),
            nn.GroupNorm(1, _in * 2),
            Activation(inplace=True),
            nn.Conv2d(_in * 2, _in * 4, 1),
            )
        self.decoder = nn.Sequential(
            nn.GroupNorm(1, _in * 4),
            Activation(inplace=True),
            nn.Conv2d(_in * 4, _out, 1),
            nn.GroupNorm(1, _out)
        )
        self.score = nn.Sequential(
            nn.GroupNorm(1, _in * 4),
            Activation(inplace=True),
            nn.Conv2d(_in * 4, _in * 4, 1, groups=_in),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor):
        x = self.encoder(x)
        s = self.score(x)
        return self.decoder(x * s)


class ReScale2D(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor, ref: Tensor):
        """
        Linearly project `x` to the radiometric scale of `y`;
        """
        scaled = (x *
                  ref.std((-1, -2), keepdim=True) /
                  x.std((-1, -2), keepdim=True).add(1e-5))
        return (scaled -
                scaled.mean((-1, -2), keepdim=True) +
                ref.mean((-1, -2), keepdim=True))
    
    def __call__(self, x: Tensor, ref: Tensor, *args: Any, **kwds: Any) -> Any:
        return super().__call__(x, ref, *args, **kwds)


class UpsamplingBlock(nn.Module):
    def __init__(self, _in: int, _out: int, size: int,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.up = nn.Upsample((size, size), mode='nearest')
        self.concat = CrossGatedConcat(_in, _out, _out // 2)
        self.prop = DualConv(_in + _out // 2, _out, stride=1)

    def forward(self, x, connection):
        x = self.up(x)
        x = self.concat(x, connection)
        return self.prop(x)


class DownsamplingBlock(nn.Module):
    def __init__(self, _in: int, _out: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.residual = ResidualProj(_in, _out, stride=2, groups=2)
        self.module = DualConv(_in, _out, stride=2, padding=1, groups=2)

    def forward(self, x):
        return self.module(x).add(self.residual(x))


class Head(nn.Module):
    def __init__(self, _in: int, _out: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.module = nn.Sequential(
            ASPP(_in, _in // 2, 1, 3, 6),
            nn.BatchNorm2d(_in // 2, momentum=CONFIG['BATCHNORM_MOMENT']),
            Activation(inplace=True),
            nn.Conv2d(_in // 2, _out, 1, padding=0),
            nn.BatchNorm2d(_out, momentum=CONFIG['BATCHNORM_MOMENT'])
        )

    def forward(self, x):
        return self.module(x)


class Bridge(nn.Module):
    def __init__(self, _in: int, _out: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.module = ASPP(_in, _out, 1, 3, 6)
        self.residual = ResidualProj(_in, _out, stride=1)

    def forward(self, x):
        return self.module(x).add(self.residual(x))


class Stem(nn.Module):
    def __init__(self, _in: int, _out: int, fuse_in: int, size: int,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embed_y = EmbeddingY(fuse_in, _in, size)
        self.embed_x = EmbeddingX(_in, _in)
        self.module = nn.Sequential(
            nn.BatchNorm2d(_in * 2, momentum=CONFIG['BATCHNORM_MOMENT']),
            nn.Conv2d(_in * 2, _out, kernel_size=3, padding=1, groups=2,
                      padding_mode=CONFIG["PADDING_MODE"]),
            nn.BatchNorm2d(_out, momentum=CONFIG['BATCHNORM_MOMENT']),
            Activation(inplace=True),
            nn.Conv2d(_out, _out, kernel_size=3, padding=1, groups=2,
                      padding_mode=CONFIG["PADDING_MODE"])
        )
        self.residual = ResidualProj(2*_in, _out, stride=1, groups=2)

    def forward(self, x, y):
        x = self.embed_x(x)
        y = self.embed_y(y)
        x = concat([x, y], dim=-3)
        return self.module(x).add(self.residual(x))


class EmbeddingY(nn.Module):
    def __init__(self, _in: int, _out: int, size: int, scale: int = 25,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.module = nn.Sequential(
             nn.BatchNorm2d(_in, momentum=CONFIG["BATCHNORM_MOMENT"]),
             nn.Upsample((size, size), mode="nearest") if size else None,
             nn.Conv2d(_in, _out, kernel_size=3, stride=1, padding=scale,
                       dilation=scale, padding_mode=CONFIG["PADDING_MODE"])
        )

    def forward(self, x):
        return self.module(x)


class EmbeddingX(nn.Module):
    def __init__(self, _in: int, _out: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.module = nn.Sequential(
             nn.InstanceNorm2d(_in, affine=True),
             nn.Conv2d(_in, _out, kernel_size=3, stride=1, padding=1,
                       padding_mode=CONFIG["PADDING_MODE"])
        )

    def forward(self, x):
        return self.module(x)


class CrossGatedConcat(nn.Module):
    def __init__(self, _in: int, _out: int, conn_in: int,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.query_x = nn.Sequential(
            nn.BatchNorm2d(_in, momentum=CONFIG['BATCHNORM_MOMENT']),
            Activation(inplace=True),
            nn.Conv2d(_in, _out,
                      kernel_size=3,
                      padding=1,
                      bias=False,
                      padding_mode=CONFIG['PADDING_MODE']))

        self.query_c = nn.Sequential(
            nn.BatchNorm2d(conn_in, momentum=CONFIG['BATCHNORM_MOMENT']),
            Activation(inplace=True),
            nn.Conv2d(conn_in, _out,
                      kernel_size=3,
                      padding=1,
                      bias=False,
                      padding_mode=CONFIG['PADDING_MODE']))
        
        self.score = nn.Sequential(
            nn.BatchNorm2d(_out, momentum=CONFIG['BATCHNORM_MOMENT']),
            Activation(inplace=True),
            nn.Conv2d(_out, _in + conn_in,
                      kernel_size=3,
                      padding=1,
                      bias=False,
                      padding_mode=CONFIG['PADDING_MODE']),
            nn.Sigmoid())
        
    def forward(self, x, connection):
        query_x = self.query_x(x)
        query_c = self.query_c(connection)
        score = self.score(query_c.add(query_x))
        return einsum("...ji,...ji->...ji",
                      concat([x, connection], dim=-3),
                      score)
    

class ASPP(nn.Module):
    def __init__(self, _in: int, _out: int, scale_1: int, scale_2: int,
                 scale_3: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scale_1 = nn.Sequential(
            nn.Conv2d(_in, _out, kernel_size=3, dilation=scale_1,
                      padding=scale_1, padding_mode=CONFIG['PADDING_MODE']),
            nn.BatchNorm2d(_out, momentum=CONFIG['BATCHNORM_MOMENT']),
            Activation(inplace=True),
        )

        self.scale_2 = nn.Sequential(
            nn.Conv2d(_in, _out, kernel_size=3, dilation=scale_2,
                      padding=scale_2, padding_mode=CONFIG['PADDING_MODE']),
            nn.BatchNorm2d(_out, momentum=CONFIG['BATCHNORM_MOMENT']),
            Activation(inplace=True),
        )

        self.scale_3 = nn.Sequential(
            nn.Conv2d(_in, _out, kernel_size=3, dilation=scale_3,
                      padding=scale_3, padding_mode=CONFIG['PADDING_MODE']),
            nn.BatchNorm2d(_out, momentum=CONFIG['BATCHNORM_MOMENT']),
            Activation(inplace=True),
        )

        self.pool = nn.Conv2d(_out, _out, kernel_size=1, padding=0)

    def forward(self, x):
        x1 = self.scale_1(x)
        x2 = self.scale_2(x)
        x3 = self.scale_3(x)
        return self.pool(x1 + x2 + x3)


class DualConv(nn.Module):
    def __init__(self, _in: int, _out: int, *, kernel_size: int = 3,
                 stride: int = 1, dilation: int = 1, padding: int = 1,
                 groups: int = 1, batch_norm_momentum: float = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.module = nn.Sequential(
            nn.BatchNorm2d(_in, momentum=batch_norm_momentum or
                           CONFIG['BATCHNORM_MOMENT']),
            Activation(inplace=True),
            nn.Conv2d(in_channels=_in, out_channels=_out,
                      kernel_size=kernel_size,
                      padding=padding,
                      stride=stride,
                      dilation=dilation,
                      padding_mode=CONFIG['PADDING_MODE'],
                      groups=groups),
            nn.BatchNorm2d(_out, momentum=batch_norm_momentum or
                           CONFIG['BATCHNORM_MOMENT']),
            Activation(inplace=True),
            nn.Conv2d(in_channels=_out, out_channels=_out,
                      kernel_size=kernel_size,
                      padding=1 if kernel_size == 3 else padding,
                      padding_mode=CONFIG['PADDING_MODE'],
                      groups=groups))
        
    def forward(self, x):
        return self.module(x)


class ResidualProj(nn.Module):
    """
    Cheap residual block that projects the data tensor to the necessary shape
    for connecting to a subsequent module.
    """
    
    def __init__(self, _in: int, _out: int, stride: int, groups: int = 1,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.module = nn.Sequential(
            nn.Conv2d(_in, _out, 1, stride=stride, groups=groups),
            nn.BatchNorm2d(_out, momentum=CONFIG['BATCHNORM_MOMENT'])
        )

    def forward(self, x):
        return self.module(x)
