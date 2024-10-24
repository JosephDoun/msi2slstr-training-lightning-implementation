from lightning import LightningModule

from metrics.fusion import msi2slstr_loss
from transformations.normalization import channel_stretch

from typing import Mapping
from typing import Any

from torch import Tensor
from torch import set_float32_matmul_precision
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim import Adam
from torch.nn import Conv2d

from math import sqrt

from .components import Stem
from .components import DownsamplingBlock
from .components import Bridge
from .components import UpsamplingBlock
from .components import Head
from .components import OpticalToThermal
from .components import Scale2D


set_float32_matmul_precision('medium')
L = msi2slstr_loss()


class msi2slstr(LightningModule):

    def __init__(self, lr: float = 1e-3, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self._extra_out = {}

    def configure_model(self) -> None:
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

        self._initialize_weights()
        return super().configure_model()

    def _initialize_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, Conv2d):
                m.weight.data.normal_(std=sqrt(2) / sqrt(m.weight.shape[1]))
                if m.bias is not None:
                    m.bias.data.fill_(.0)

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
        loss = L(x, Y_hat, y, self._extra_out['thermal_y'],
                 self._extra_out['thermal_x'])
        # Channelwise evaluation.
        energy = L.energy(y, Y_hat).mean(0)
        thermal = L.evaluate(y[:, 6:], self._extra_out['thermal_y']).mean(0)

        self._extra_out['x'] = x
        self._extra_out['y'] = y
        self._extra_out['Y_hat'] = Y_hat

        batch_loss = loss.mean()
        # Batchwise evaluation.
        sample_loss = loss.detach().mean(-1)

        self.log("loss/train", batch_loss,
                 logger=True, prog_bar=True, on_epoch=True,
                 on_step=True, batch_size=sample_loss.size(0))

        self.log_dict({**{f"thermal_{i}/train": v for i, v in
                          enumerate(thermal)},
                       # Per date batch.
                       **{f"{k}/train": v for k, v in zip(dates, sample_loss,
                                                          strict=True)},
                       # Per tile.
                       **{f"{k}/train": v for k, v in zip(tiles, sample_loss,
                                                          strict=True)}
                       },
                      on_step=True,
                      on_epoch=True,
                      prog_bar=False,
                      logger=True,
                      batch_size=sample_loss.size(0))

        return batch_loss
    
    def on_train_epoch_end(self) -> None:
        tboard: SummaryWriter = self.logger.experiment

        tboard.add_images(tag="x/train",
                          img_tensor=channel_stretch(self._extra_out['x'][0])
                                                     .unsqueeze(1),
                          global_step=self.current_epoch,
                          dataformats='NCHW')

        tboard.add_images(tag="y/train",
                          img_tensor=channel_stretch(self._extra_out['y'][0])
                                                     .unsqueeze(1),
                          global_step=self.current_epoch,
                          dataformats='NCHW')

        tboard.add_images(tag="Y_hat/train",
                          img_tensor=channel_stretch(self._extra_out['Y_hat']
                                                     [0]).unsqueeze(1),
                          global_step=self.current_epoch,
                          dataformats='NCHW')
        

    def validation_step(self, batch, batch_idx) -> Tensor | Mapping[str, Any] | None:
        data, metadata = batch
        dates, tiles = metadata
        x, y = data

        Y_hat = self(x, y)
        loss = L(x, Y_hat, y, self._extra_out['thermal_y'],
                 self._extra_out['thermal_x'])
        # Channelwise evaluation.
        energy = L.energy(y, Y_hat).mean(0)
        thermal = L.evaluate(y[:, 6:], self._extra_out['thermal_y']).mean(0)

        batch_loss = loss.mean()
        sample_loss = loss.detach().mean(-1)

        self.log("loss/val", batch_loss, prog_bar=True, logger=True,
                 on_epoch=True, on_step=True, batch_size=sample_loss.size(0))
        
        self.log_dict({**{f"thermal_{i}/val": v
                          for i, v in enumerate(thermal)},
                       # Per date.
                       **{f"{k}/val": v for k, v in zip(dates, sample_loss,
                                                        strict=True)},
                       # Per tile.
                       **{f"{k}/val": v for k, v in zip(tiles, sample_loss,
                                                        strict=True)}
                       },
                      on_step=False,
                      on_epoch=True,
                      prog_bar=False,
                      logger=True,
                      batch_size=sample_loss.size(0))

        return batch_loss
    
    def test_step(self, batch, batch_idx) -> Tensor | Mapping[str, Any] | None:
        data, metadata = batch
        dates, tiles = metadata
        x, y = data
        Y_hat = self(x, y)
        
        loss = L(x, Y_hat, y, self._extra_out['thermal_y'],
                 self._extra_out['thermal_x'])
        
        # Channelwise evaluation.
        energy = L.energy(y, Y_hat).mean(0)
        thermal = L.evaluate(y[:, 6:], self._extra_out['thermal_y']).mean(0)
        
        batch_loss = loss.mean()
        sample_loss = loss.detach().mean(-1)

        self.log("loss/test", batch_loss, prog_bar=True, logger=True,
                 on_step=True, on_epoch=True, batch_size=sample_loss.size(0))
        
        self.log_dict({**{f"thermal_{i}/test": v
                          for i, v in enumerate(thermal)},
                       # Per date.
                       **{f"{k}/test": v for k, v in zip(dates, sample_loss,
                                                         strict=True)},
                       # Per tile.
                       **{f"{k}/test": v for k, v in zip(tiles, sample_loss,
                                                         strict=True)}
                       },
                      on_step=True,
                      on_epoch=True,
                      prog_bar=False,
                      logger=True,
                      batch_size=sample_loss.size(0))

        return batch_loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr, maximize=True)
