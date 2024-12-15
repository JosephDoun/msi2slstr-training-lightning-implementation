from typing import Any

from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim import Adam

from lightning import LightningModule

from transformations.normalization import channel_stretch

from .components import StaticNorm2D
from .components import ReflectedToEmitted

from metrics import ssim


class emissivity_module(LightningModule):
    def __init__(self, lr=1e-3, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.norm = StaticNorm2D("sen3")
        self.module = ReflectedToEmitted(6, 6)
        self._extra_out = {}
        self.save_hyperparameters()

        self.loss = ssim(a=1, b=.5, c=3)

    def forward(self, x):
        x = self.norm(x)[:, :6]
        return self.norm.denorm(self.module(x), slice(6, 13))

    def training_step(self, batch, batch_idx) -> Tensor:
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

    def validation_step(self, batch, batch_idx) -> Tensor:
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

    def test_step(self, batch, batch_idx) -> Tensor:
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
