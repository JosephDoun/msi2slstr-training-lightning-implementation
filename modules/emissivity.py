from typing import Any, Callable

from torch import randint
from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim import Optimizer

from torch.nn import Conv2d
from torch import arange

from lightning import LightningModule

from transformations.normalization import channel_stretch
from transformations.resampling import NonStrictAvgDownSamplingModule as Down

from .components import StaticNorm2D
from .components import ReflectiveToEmissive

from metrics.ssim import ssim3d

from math import sqrt


class emissivity_module(LightningModule):
    """
    Make use of the corregistered Sentinel-2 images
    for the estimation of emissivity in pixels.
    """
    def __init__(self, lr=1e-3, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.xnorm = StaticNorm2D("sen2")
        self.ynorm = StaticNorm2D("sen3")
        self._extra_out = {}

        self.module = ReflectiveToEmissive(13, 12)
        self.save_hyperparameters()

        self._loss = ssim3d(a=1, b=1, c=1, agg='mean')

    def _initialize_weights(self, m):
        if isinstance(m, Conv2d) and m.weight.requires_grad:
            m.weight.data.normal_(std=sqrt(2) / sqrt(m.weight.shape[1]))
            if m.bias is not None:
                m.bias.data.normal_(std=.01)

    def configure_model(self) -> None:
        self.apply(self._initialize_weights)
        return super().configure_model()
    
    def forward(self, x):
        return self.module(x)

    def training_step(self, batch, batch_idx) -> Tensor:
        """
        Expects corregistered pairs of Sentinel-2 and Sentinel-3 patches.
        """
        # Unpack data.
        data, _ = batch
        x, y = data
        x, y = self.xnorm(x), self.ynorm(y)

        # Directly use the 6 mutual regions of spectrum
        # for training. Avoids the need for radiometric scaling
        # of the Sentinel-2 input later.
        # Update: Use whole Sentinel-2 input to target emissivity bands.
        x = Down(x)
        
        # Randomly dropout a single input band.
        x[arange(x.size(0)), randint(13, size=(x.size(0),))] = 0

        # Use directly all corregistered Sentinel-2 bands
        # to estimate thermal emissivity. See above.
        Y_hat = self(x)
        
        # Validate against target thermal emissivity.
        loss = self._loss(Y_hat, y)

        # Log.
        per_band = loss.mean(0)
        batch_loss = loss.mean()

        self._extra_out["y"] = y[batch_loss.mean(-1).argmax()]
        self._extra_out["Y_hat"] = Y_hat[batch_loss.mean(-1).argmax()]

        self.log("hp_metric", batch_loss, batch_size=loss.size(0))
        self.log("emissivity/train/loss",
                 batch_loss,
                 batch_size=loss.size(0),
                 on_step=True, on_epoch=True,
                 prog_bar=True)

        self.log_dict({f"emissivity/train_{i}": v for i, v in
                       enumerate(per_band)}, batch_size=loss.size(0),
                       on_step=True)
        return batch_loss

    def on_train_epoch_end(self) -> None:
        tboard: SummaryWriter = self.logger.experiment

        tboard.add_images(tag="emissivity/train/y",
                          img_tensor=channel_stretch(self._extra_out['y'])
                                                     .unsqueeze(1),
                          global_step=self.current_epoch,
                          dataformats='NCHW')

        tboard.add_images(tag="emissivity/train/Y_hat",
                          img_tensor=channel_stretch(self._extra_out['Y_hat'])
                                                     .unsqueeze(1),
                          global_step=self.current_epoch,
                          dataformats='NCHW')

    def validation_step(self, batch, batch_idx) -> Tensor:
        """
        Expects corregistered pairs of Sentinel-2 and Sentinel-3 patches.
        """
        # Unpack data.
        data, _ = batch
        x, y = data
        x, y = self.xnorm(x), self.ynorm(y)

        x = Down(x)

        # Use directly all corregistered Sentinel-2 bands
        # to estimate thermal emissivity.
        Y_hat = self(x)
        
        # Validate against real thermal emissivity.
        loss = self._loss(Y_hat, y)

        # Log.
        per_band = loss.mean(0)
        batch_loss = loss.mean()

        self.log("emissivity/valid/loss",
                 batch_loss,
                 batch_size=loss.size(0),
                 on_step=True, on_epoch=True,
                 prog_bar=True)

        self.log_dict({f"emissivity/valid_{i}": v for i, v in
                       enumerate(per_band)}, batch_size=loss.size(0),
                       on_step=True)

        return batch_loss

    def configure_optimizers(self):
        opt = Adam(self.parameters(), lr=self.hparams.lr, maximize=True)
        sch = CosineAnnealingWarmRestarts(opt, T_0=100, T_mult=2, eta_min=1e-5)
        return [opt], [{"scheduler": sch, "interval": "step",
                        "name": "lr/emis/cosine"}]

    def optimizer_step(self, epoch: int, batch_idx: int,
                       optimizer: Optimizer,
                       optimizer_closure: Callable[[], Any] | None = None) -> None:
        
        # skip the first 1000 steps
        if self.trainer.global_step < 500:
            lr_scale = min(1.0, self.trainer.global_step >= 499 or 0)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr
        
        return super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)