from typing import Any
from lightning import LightningModule
from metrics.fusion import msi2slstr_loss
from torch import Tensor
from torch import set_float32_matmul_precision
from typing import Mapping

from .components import Stem
from .components import DownsamplingBlock
from .components import Bridge
from .components import UpsamplingBlock
from .components import Head
from .components import OpticalToThermal
from .components import Scale2D

from torch.optim import Adam


set_float32_matmul_precision('medium')


class msi2slstr(LightningModule):

    def __init__(self, lr: float = 1e-3, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.loss = msi2slstr_loss()
        self.scale = Scale2D()
        self.therm = OpticalToThermal(6, 6)
        self.stem = Stem(13, 32, 12, 100)
        self.down_a = DownsamplingBlock( 32,  64)
        self.down_b = DownsamplingBlock( 64, 128)
        self.down_c = DownsamplingBlock(128, 256)
        self.bridge = Bridge(256, 512) # 13
        self.up_c = UpsamplingBlock(512, 256,  25)
        self.up_b = UpsamplingBlock(256, 128,  50)
        self.up_a = UpsamplingBlock(128,  64, 100)
        self.head = Head(64, 12)

        self._extra_out = {}

    def forward(self, x, y) -> Tensor:
        optic_y = y[:, :6]
        optic_x = self.scale(x[:, [2, 3, 8, 10, 11, 12]], optic_y)

        # Only needed during training.
        self._extra_out['thermal_y'] = self.therm(optic_y)
        
        # This serves only as target.
        self._extra_out['thermal_x'] = self.therm(optic_x).detach()

        a = self.stem(x, y)
        b = self.down_a(a)
        c = self.down_b(b)
        x = self.down_c(c)
        x = self.bridge(x)
        x = self.up_c(x, c)
        x = self.up_b(x, b)
        x = self.up_a(x, a)
        return self.head(x)

    def training_step(self, batch, batch_idx) -> Tensor | Mapping[str, Any] | None:
        data, metadata = batch
        dates, tiles = metadata
        x, y = data

        Y_hat = self(x, y)
        loss = self.loss(x, Y_hat, y,
                         self._extra_out['thermal_y'],
                         self._extra_out['thermal_x'])

        batch_loss = loss.mean()
        sample_loss = loss.mean(-1)

        # self.logger.experiment.add_scalars
        self.log_dict({"training_loss": batch_loss,
                       # Per date.
                       **{f"{k}/train": v for k, v in zip(dates, sample_loss,
                                                          strict=True)},
                       # Per tile.
                       **{f"{k}/train": v for k, v in zip(tiles, sample_loss,
                                                          strict=True)}
                       },
                      on_step=False,
                      on_epoch=True,
                      prog_bar=True,
                      logger=True,
                      batch_size=sample_loss.size(0))

        return batch_loss
    
    def validation_step(self, batch, batch_idx) -> Tensor | Mapping[str, Any] | None:
        data, metadata = batch
        dates, tiles = metadata
        x, y = data
        Y_hat = self(x, y)
        loss = self.loss(x, Y_hat, y,
                         self._extra_out['thermal_y'],
                         self._extra_out['thermal_x'])

        batch_loss = loss.mean()
        sample_loss = loss.mean(-1)
        
        self.log_dict({"validation_loss": batch_loss,
                       # Per date.
                       **{f"{k}/val": v for k, v in zip(dates, sample_loss,
                                                        strict=True)},
                       # Per tile.
                       **{f"{k}/val": v for k, v in zip(tiles, sample_loss,
                                                        strict=True)}
                       },
                      on_step=False,
                      on_epoch=True,
                      prog_bar=True,
                      logger=True,
                      batch_size=sample_loss.size(0))
        
        return batch_loss
    
    def test_step(self, batch, batch_idx) -> Tensor | Mapping[str, Any] | None:
        data, metadata = batch
        dates, tiles = metadata
        x, y = data
        Y_hat = self(x, y)
        loss = self.loss(x, Y_hat, y,
                         self._extra_out['thermal_y'],
                         self._extra_out['thermal_x'])

        batch_loss = loss.mean()
        sample_loss = loss.mean(-1)

        self.log_dict({"test_loss": batch_loss,
                       # Per date.
                       **{f"{k}/test": v for k, v in zip(dates, sample_loss,
                                                         strict=True)},
                       # Per tile.
                       **{f"{k}/test": v for k, v in zip(tiles, sample_loss,
                                                         strict=True)}
                       },
                      on_step=False,
                      on_epoch=True,
                      prog_bar=True,
                      logger=True,
                      batch_size=sample_loss.size(0))

        return batch_loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr, maximize=True)
