from lightning import LightningModule
from torch.optim.optimizer import Optimizer

from metrics.fusion import msi2slstr_loss
from metrics.ssim import ssim
from metrics.latent import DeepLoss

from transformations.normalization import channel_stretch

from typing import Mapping
from typing import Any

from torch import Tensor
from torch import no_grad
from torch import concat
from torch import set_float32_matmul_precision
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.nn import Conv2d
from torch.nn import MSELoss
from torch.nn import CosineSimilarity

from math import sqrt

from .components import StaticNorm2D
from .components import Stem
from .components import DownsamplingBlock
from .components import Bridge
from .components import UpsamplingBlock
from .components import Head
from .components import OpticalToThermal
from .components import ReScale2D

from config import DATA_CONFIG


set_float32_matmul_precision('high')

MSI2SLSTRLoss = msi2slstr_loss(a=3)
SSIM = ssim(c=3)
DEEPLOSS = DeepLoss([512], [13],
                    loss_fn=MSELoss(reduction="none"),
                    maximize=False)

from torch.nn.utils import clip_grad_norm_


class msi2slstr(LightningModule):

    def __init__(self, lr: float = 1e-3, size: int = 100,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        assert not size % 50, "Size not multiple of 50."
        self.save_hyperparameters()
        self._extra_out = {}
        self.xnorm = StaticNorm2D("sen2")
        self.ynorm = StaticNorm2D("sen3")
        self.stem = Stem(13, 32, 12, size)
        self.rescale = ReScale2D()
        self.therm = OpticalToThermal(6, 6)
        self.down_a = DownsamplingBlock( 32,  64)
        self.down_b = DownsamplingBlock( 64, 128)
        self.down_c = DownsamplingBlock(128, 256)
        self.bridge = Bridge(256, 512) # 13x13
        self.up_c = UpsamplingBlock(512, 256, size // 4)
        self.up_b = UpsamplingBlock(256, 128, size // 2)
        self.up_a = UpsamplingBlock(128,  64, size)
        self.head = Head(64, 12)
        self._initialize_weights()

        self.loss = msi2slstr_loss(a=2, b=1, c=1)
        self.thermloss = ssim(a=1, b=.5, c=3)

    def _initialize_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, Conv2d) and m.weight.requires_grad:
                m.weight.data.normal_(std=sqrt(2) / sqrt(m.weight.shape[1]))
                if m.bias is not None:
                    m.bias.data.fill_(.0)

    def forward(self, x, y) -> Tensor:
        x = self.xnorm(x)
        y = self.ynorm(y)
        a = self.stem(x, y)

        optic_y = y[:, [0, 1, 2, 3, 4, 5]]

        # x scaled to value range of y.
        optic_x = self.rescale(x=x[:, DATA_CONFIG['sen2_bands']],
                               ref=optic_y)

        self._extra_out['thermal_y'] = self.ynorm.denorm(
            self.therm(optic_y), channels=slice(6, 13, 1))

        # This serves only as target.
        # Eval not necessary as stats guaranteed to be same.
        with no_grad():
            self._extra_out['thermal_x'] = self.ynorm.denorm(
                self.therm(optic_x).detach(), channels=slice(6, 13, 1))

        b = self.down_a(a)
        self._extra_out['deep64'] = b

        c = self.down_b(b)
        self._extra_out['deep128'] = c

        x = self.down_c(c)
        self._extra_out['deep256'] = x

        # The output of the bridge should be targeted to y directly.
        x = self.bridge(x)
        self._extra_out['deep512'] = x

        x = self.up_c(x, c)
        x = self.up_b(x, b)
        x = self.up_a(x, a)
        x = self.head(x)
        return self.ynorm.denorm(x)

    def training_step(self, batch, batch_idx) -> Tensor | Mapping[str, Any]:
        data, metadata = batch
        dates, tiles = metadata
        x, y = data

        Y_hat = self(x, y)

        loss = self.loss(x, Y_hat, y, self._extra_out['thermal_y'],
                         self._extra_out['thermal_x'])

        thermal = self.thermloss(self._extra_out['thermal_y'], y[:, 6:])

        # For channelwise evaluation.
        energy = loss.mean(0)
        thermal = thermal.mean(0)

        self._extra_out['x'] = x
        self._extra_out['y'] = y
        self._extra_out['Y_hat'] = Y_hat

        batch_loss = energy.mean()

        # For samplewise evaluation.
        sample_loss = loss.mean(-1)

        self.log("hp_metric", batch_loss)
        self.log("training/loss/train", batch_loss,
                 logger=True, prog_bar=True, on_epoch=True,
                 on_step=True, batch_size=sample_loss.size(0))

        self.log_dict({**{f"training/energy_{i}/train": v for i, v in
                          enumerate(energy)},

                       **{f"training/thermal_{i}/train": v for i, v in
                          enumerate(thermal)},

                       # Per date batch.
                       **{f"training/{k}/train": v for k, v in
                          zip(dates, sample_loss, strict=True)},

                       # Per tile.
                       **{f"training/{k}/train": v for k, v in
                          zip(tiles, sample_loss, strict=True)}
                       },
                      on_step=True,
                      on_epoch=True,
                      prog_bar=False,
                      logger=True,
                      batch_size=sample_loss.size(0))

        deep_loss = DEEPLOSS(self._extra_out['deep512'],
                             y=y).mean()

        return batch_loss.add(deep_loss).add(thermal.mean())

    def on_train_epoch_end(self) -> None:
        tboard: SummaryWriter = self.logger.experiment

        tboard.add_images(tag="training/x/train",
                          img_tensor=channel_stretch(self._extra_out['x'][1])
                                                     .unsqueeze(1),
                          global_step=self.current_epoch,
                          dataformats='NCHW')

        tboard.add_images(tag="training/y/train",
                          img_tensor=channel_stretch(self._extra_out['y'][1])
                                                     .unsqueeze(1),
                          global_step=self.current_epoch,
                          dataformats='NCHW')

        tboard.add_images(tag="training/y/thermal",
                          img_tensor=channel_stretch(self._extra_out
                                                     ['thermal_y'][1])
                                                     .unsqueeze(1),
                          global_step=self.current_epoch,
                          dataformats='NCHW')

        tboard.add_images(tag="training/Y_hat/train",
                          img_tensor=channel_stretch(self._extra_out['Y_hat']
                                                     [1]).unsqueeze(1),
                          global_step=self.current_epoch,
                          dataformats='NCHW')

    def validation_step(self, batch, batch_idx) -> Tensor | Mapping[str, Any]:
        data, metadata = batch
        dates, tiles = metadata
        x, y = data

        Y_hat = self(x, y)

        loss = self.loss(x, Y_hat, y,
                                  self._extra_out['thermal_y'],
                                  self._extra_out['thermal_x'])
        
        thermal = self.thermloss(self._extra_out['thermal_y'], y[:, 6:])

        # For channelwise evaluation.
        energy = loss.mean(0)
        thermal = thermal.mean(0)

        batch_loss = loss.mean()
        sample_loss = loss.detach().mean(-1)

        self.log("training/loss/val", batch_loss, prog_bar=True, on_epoch=True,
                 batch_size=sample_loss.size(0))

        self.log_dict({**{f"training/energy_{i}/val": v for i, v in
                          enumerate(energy)},

                       **{f"training/thermal_{i}/val": v
                          for i, v in enumerate(thermal)},

                       # Per date.
                       **{f"training/{k}/val": v for k, v in
                          zip(dates, sample_loss, strict=True)},

                       # Per tile.
                       **{f"training/{k}/val": v for k, v in
                          zip(tiles, sample_loss, strict=True)}
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

        loss = self.loss(x, Y_hat, y,
                                  self._extra_out['thermal_y'],
                                  self._extra_out['thermal_x'])
        thermal = self.thermloss(self._extra_out['thermal_y'], y[:, 6:])

        # For channelwise evaluation.
        energy = loss.mean(0)
        thermal = thermal.mean(0)

        batch_loss = loss.mean()
        sample_loss = loss.detach().mean(-1)

        self.log("training/loss/test", batch_loss, prog_bar=True, logger=True,
                 on_step=True, on_epoch=True, batch_size=sample_loss.size(0))
        
        self.log_dict({**{f"training/energy_{i}/test": v for i, v in
                          enumerate(energy)},

                       **{f"training/thermal_{i}/test": v
                          for i, v in enumerate(thermal)},

                       # Per date.
                       **{f"training/{k}/test": v for k, v in
                          zip(dates, sample_loss, strict=True)},

                       # Per tile.
                       **{f"training/{k}/test": v for k, v in
                          zip(tiles, sample_loss, strict=True)}
                       },
                      on_step=True,
                      on_epoch=True,
                      prog_bar=False,
                      logger=True,
                      batch_size=sample_loss.size(0))

        return batch_loss

    def predict_step(self, batch, batch_idx) -> None:
        indices, (x, y) = batch

        # Predict
        Y_hat = self(x, y)

        # Write to output: A writer class should be available to receive the
        # inference product.
        self.trainer.predict_dataloaders.dataset.output(indices, Y_hat)

    def configure_optimizers(self):
        opt = Adam(self.parameters(), lr=self.hparams.lr, maximize=True,
                    betas=(.9, .92), weight_decay=1e-10, eps=1e-6)
        sch = CosineAnnealingWarmRestarts(opt, T_0=100, T_mult=2,
                                          eta_min=1e-8)
        return [opt], [{"scheduler": sch, "interval": "step"}]


class msi2slstr_pretraining(msi2slstr):

    def __init__(self, lr: float = 0.001, *args: Any, **kwargs: Any) -> None:
        super().__init__(lr, *args, **kwargs)
        self.therm = thermal_prediction.load_from_checkpoint(
            "pretrained/thermal.ckpt").module

    def training_step(self, batch, batch_idx) -> Tensor | Mapping[str, Any]:
        (x, y), _ = batch

        for i in range(13):
            x[:, [i]].normal_(self.xnorm.mean.squeeze()[i].item(),
                              self.xnorm.var.squeeze()[i].sqrt().item())

        x.normal_()
        Y_hat = self(x, y)

        self._extra_out['y'] = y
        self._extra_out['Y_hat'] = Y_hat

        # Main loss.
        loss = SSIM(MSI2SLSTRLoss._usample(y), Y_hat)
        loss2 = SSIM.s(concat([x[:, [2,3,8,10,11,12]],
                               self._extra_out["thermal_x"]], dim=1),
                               Y_hat)

        # For the OptToThermal optimization.
        thermal = SSIM(y[:, 6:], self._extra_out['thermal_y']).mean(0)

        # Channelwise evaluation.
        bands_loss = loss.mean(0)
        batch_loss = loss.mean()

        self.log("hp_metric", batch_loss)
        self.log("pretraining/train_loss", batch_loss,
                 logger=True, prog_bar=True, on_epoch=True,
                 on_step=True, batch_size=loss.size(0))

        self.log_dict({**{f"pretraining/energy_{i}/train": v for i, v in
                          enumerate(bands_loss)},
                       **{f"pretraining/thermal_{i}/train": v for i, v in
                          enumerate(thermal)},
                       },
                      on_step=True,
                      on_epoch=True,
                      prog_bar=False,
                      logger=True,
                      batch_size=loss.size(0))

        deep_loss = DEEPLOSS(self._extra_out['deep512'],
                             y=y).mean()

        return batch_loss + deep_loss + thermal.mean() + loss2.mean()

    def on_train_epoch_end(self) -> None:
        tboard: SummaryWriter = self.logger.experiment

        tboard.add_images(tag="pretraining/train/y",
                          img_tensor=channel_stretch(self._extra_out['y'][1])
                                                     .unsqueeze(1),
                          global_step=self.current_epoch,
                          dataformats='NCHW')

        tboard.add_images(tag="pretraining/train/Y_hat",
                          img_tensor=channel_stretch(self._extra_out['Y_hat']
                                                     [1]).unsqueeze(1),
                          global_step=self.current_epoch,
                          dataformats='NCHW')

        tboard.add_images(tag="pretraining/train/thermal",
                          img_tensor=channel_stretch(self._extra_out
                                                     ['thermal_y']
                                                     [1]).unsqueeze(1),
                          global_step=self.current_epoch,
                          dataformats='NCHW')

    def validation_step(self, batch, batch_idx) -> Tensor | Mapping[str, Any]:
        return None

    def test_step(self, batch, batch_idx) -> Tensor | Mapping[str, Any]:
        return None


class thermal_prediction(LightningModule):
    def __init__(self, lr=1e-3, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.norm = StaticNorm2D("sen3")
        self.module = OpticalToThermal(6, 6)
        self._extra_out = {}
        self.save_hyperparameters()

        self.loss = ssim(a=1, b=.5, c=3)

    def forward(self, x):
        x = self.norm(x)[:, :6]
        return self.norm.denorm(self.module(x), slice(6, 13))

    def training_step(self, batch, batch_idx) -> Tensor | Mapping[str, Any]:
        Y_hat = self(batch)
        loss = self.loss(batch[:, 6:], Y_hat)
        band_loss = loss.mean(0)
        batch_loss = loss.mean()

        self._extra_out["y"] = batch[batch_loss.mean(-1).argmax()]
        self._extra_out["Y_hat"] = Y_hat[batch_loss.mean(-1).argmax()]

        self.log("hp_metric", batch_loss, batch_size=band_loss.size(0))
        self.log("opt_to_thermal_experiment/train/loss",
                 batch_loss,
                 batch_size=batch.size(0),
                 on_step=True, prog_bar=True)

        self.log_dict({f"opt_to_thermal_experiment/train_{i}": v for i, v in
                       enumerate(band_loss)}, batch_size=batch.size(0),
                       on_step=True)

        return batch_loss

    def on_train_epoch_end(self) -> None:
        tboard: SummaryWriter = self.logger.experiment

        tboard.add_images(tag="opt_to_thermal_experiment/train/y",
                          img_tensor=channel_stretch(self._extra_out['y'])
                                                     .unsqueeze(1),
                          global_step=self.current_epoch,
                          dataformats='NCHW')

        tboard.add_images(tag="opt_to_thermal_experiment/train/Y_hat",
                          img_tensor=channel_stretch(self._extra_out['Y_hat'])
                                                     .unsqueeze(1),
                          global_step=self.current_epoch,
                          dataformats='NCHW')

    def validation_step(self, batch, batch_idx) -> Tensor | Mapping[str, Any]:
        Y_hat = self(batch)
        loss = self.loss(batch[:, 6:], Y_hat)
        band_loss = loss.mean(0)
        batch_loss = loss.mean()

        self.log("opt_to_thermal_experiment/val/loss",
                 batch_loss,
                 batch_size=batch.size(0), on_step=True, prog_bar=True)

        self.log_dict({f"opt_to_thermal_experiment/val_{i}": v for i, v in
                       enumerate(band_loss)}, batch_size=batch.size(0),
                       on_step=True)

        return batch_loss

    def test_step(self, batch, batch_idx) -> Tensor | Mapping[str, Any]:
        Y_hat = self(batch)
        loss = self.loss(batch[:, 6:], Y_hat)
        band_loss = loss.mean(0)
        batch_loss = loss.mean()

        self.log("opt_to_thermal_experiment/test/loss",
                 batch_loss,
                 batch_size=batch.size(0), on_step=True, prog_bar=True)

        self.log_dict({f"opt_to_thermal_experiment/test_{i}": v for i, v in
                       enumerate(band_loss)}, batch_size=batch.size(0),
                       on_step=True)

        return batch_loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr, maximize=True)


class msi2slstr_debug(msi2slstr):
    def configure_gradient_clipping(self, optimizer: Optimizer,
                                    gradient_clip_val: float = None,
                                    gradient_clip_algorithm: str = None) -> None:
        for n, p in self.named_parameters():
            self.print(n, p.grad.abs().mean())
            ...
        norm = clip_grad_norm_(self.parameters(), max_norm=gradient_clip_val)
        self.print("Gradient norm: ", norm)
        return super().configure_gradient_clipping(optimizer,
                                                   gradient_clip_val,
                                                   gradient_clip_algorithm)


class msi2slstr_predict(msi2slstr):
    def __init__(self, lr: float = 0.001, size: int = 100,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(lr, size, *args, **kwargs)

    def predict_step(self, batch, batch_idx) -> None:
        return super().predict_step(batch, batch_idx)