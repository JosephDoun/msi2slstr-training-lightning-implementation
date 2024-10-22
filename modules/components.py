import torch.nn as nn

from torch import einsum
from torch import concat

from . import CONFIG


class UpsamplingBlock(nn.Module):
    def __init__(self, _in: int, _out: int, size: int, *args, **kwargs) -> None:
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
        return self.module(x) + self.residual(x)


class Head(nn.Module):
    def __init__(self, _in: int, _out: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.module = nn.Sequential(
            ASPP(_in, _in // 2, 1, 3, 6),
            nn.BatchNorm2d(_in // 2, momentum=CONFIG['BATCHNORM_MOMENT']),
            nn.ReLU(True),
            nn.Conv2d(_in // 2, _out, 1, padding=0),
            nn.Softplus(beta=.2, threshold=.1)
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
        self.embed = Embedding(fuse_in, _in, size)
        self.module = nn.Sequential(
            nn.BatchNorm2d(_in * 2, momentum=CONFIG['BATCHNORM_MOMENT']),
            nn.Conv2d(_in * 2, _out, kernel_size=3, padding=1, groups=2,
                      padding_mode=CONFIG["PADDING_MODE"]),
            nn.BatchNorm2d(_out, momentum=CONFIG['BATCHNORM_MOMENT']),
            nn.ReLU(inplace=True),
            nn.Conv2d(_out, _out, kernel_size=3, padding=1, groups=2,
                      padding_mode=CONFIG["PADDING_MODE"])
        )
        self.residual = ResidualProj(2*_in, _out, stride=1, groups=2)

    def forward(self, x, y):
        y = self.embed(y)
        x = concat([x, y], dim=-3)
        return self.module(x).add(self.residual(x))


class Embedding(nn.Module):
    def __init__(self, _in: int, _out: int, size: int,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.module = nn.Sequential(
            nn.Upsample((size, size), mode="nearest"),
            nn.BatchNorm2d(_in, momentum=CONFIG["BATCHNORM_MOMENT"]),
            nn.Conv2d(_in, _out, kernel_size=3, stride=1, padding=25,
                      dilation=25, padding_mode=CONFIG["PADDING_MODE"])
        )

    def forward(self, x):
        return self.module(x)


class CrossGatedConcat(nn.Module):
    def __init__(self, _in: int, _out: int, conn_in: int,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.query_x = nn.Sequential(
            nn.BatchNorm2d(_in, momentum=CONFIG['BATCHNORM_MOMENT']),
            nn.ReLU(inplace=True),
            nn.Conv2d(_in, _out,
                      kernel_size=3,
                      padding=1,
                      bias=False,
                      padding_mode=CONFIG['PADDING_MODE']))

        self.query_c = nn.Sequential(
            nn.BatchNorm2d(conn_in, momentum=CONFIG['BATCHNORM_MOMENT']),
            nn.ReLU(inplace=True),
            nn.Conv2d(conn_in, _out,
                      kernel_size=3,
                      padding=1,
                      bias=False,
                      padding_mode=CONFIG['PADDING_MODE']))
        
        self.score = nn.Sequential(
            nn.BatchNorm2d(_out, momentum=CONFIG['BATCHNORM_MOMENT']),
            nn.ReLU(inplace=True),
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
            nn.ReLU(inplace=True)
        )

        self.scale_2 = nn.Sequential(
            nn.Conv2d(_in, _out, kernel_size=3, dilation=scale_2,
                      padding=scale_2, padding_mode=CONFIG['PADDING_MODE']),
            nn.BatchNorm2d(_out, momentum=CONFIG['BATCHNORM_MOMENT']),
            nn.ReLU(inplace=True)
        )

        self.scale_3 = nn.Sequential(
            nn.Conv2d(_in, _out, kernel_size=3, dilation=scale_3,
                      padding=scale_3, padding_mode=CONFIG['PADDING_MODE']),
            nn.BatchNorm2d(_out, momentum=CONFIG['BATCHNORM_MOMENT']),
            nn.ReLU(inplace=True)
        )

        self.pool = nn.Conv2d(_out, _out, kernel_size=1, padding=0)

    def forward(self, x):
        x1 = self.scale_1(x)
        x2 = self.scale_2(x)
        x3 = self.scale_3(x)
        return self.pool(x1 + x2 + x3)


class DualConv(nn.Module):
    def __init__(self, _in: int, _out: int, stride: int, dilation: int = 1,
                 padding: int = 1, groups: int = 1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.module = nn.Sequential(
            nn.BatchNorm2d(_in, momentum=CONFIG['BATCHNORM_MOMENT']),
            nn.ReLU(inplace=True),
            nn.Conv2d(_in, _out,
                      kernel_size=3,
                      padding=padding,
                      stride=stride,
                      dilation=dilation,
                      padding_mode=CONFIG['PADDING_MODE'],
                      groups=groups),
            nn.BatchNorm2d(_out, momentum=CONFIG['BATCHNORM_MOMENT']),
            nn.ReLU(inplace=True),
            nn.Conv2d(_out, _out,
                      kernel_size=3,
                      padding=1,
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
