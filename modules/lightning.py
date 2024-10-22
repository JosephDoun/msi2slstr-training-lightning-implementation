from typing import Any
from lightning import LightningModule

from torch import Tensor
from typing import Mapping

from .components import Stem
from .components import DownsamplingBlock
from .components import Bridge
from .components import UpsamplingBlock
from .components import Head


class msi2slstr(LightningModule):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.stem = Stem(13, 32, 12, 100)
        self.down_a = DownsamplingBlock( 32,  64)
        self.down_b = DownsamplingBlock( 64, 128)
        self.down_c = DownsamplingBlock(128, 256)
        self.bridge = Bridge(256, 512) # 13
        self.up_c = UpsamplingBlock(512, 256,  25)
        self.up_b = UpsamplingBlock(256, 128,  50)
        self.up_a = UpsamplingBlock(128,  64, 100)
        self.head = Head(64, 12)

    def forward(self, x, y) -> Tensor:
        a = self.stem(x, y)
        b = self.down_a(a)
        c = self.down_b(b)
        x = self.down_c(c)
        x = self.bridge(x)
        x = self.up_c(x, c)
        x = self.up_b(x, b)
        x = self.up_a(x, a)
        return self.head(x)

    def training_step(self, *args: Any, **kwargs: Any) -> Tensor | Mapping[str, Any] | None:
        return super().training_step(*args, **kwargs)
    
    def validation_step(self, *args: Any, **kwargs: Any) -> Tensor | Mapping[str, Any] | None:
        return super().validation_step(*args, **kwargs)