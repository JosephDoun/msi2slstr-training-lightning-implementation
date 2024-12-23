"""
Experiment 1:

Describe the desired fusion result through the definition of a fusion loss
function:

Concept:
a. SSIM loss for the downsampled output
b. Correlation with the high resolution input at full scale.
"""

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
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.nn import Conv2d
from torch.nn import AvgPool2d
from torch.nn import MSELoss

from math import sqrt

from .components import StaticNorm2D
from .components import FusionStem
from .components import DownsamplingBlock
from .components import Bridge
from .components import FusionUpsamplingBlock
from .components import Head
from .components import ReflectiveToEmissive
from .components import ReScale2D

from .emissivity import emissivity_module

from config import DATA_CONFIG


DEEPLOSS = DeepLoss([512], [13],
                    loss_fn=MSELoss(reduction="none"),
                    maximize=False)

from torch.nn.utils import clip_grad_norm_


class msi2slstr(LightningModule):

    def __init__(self, lr: float = 1e-3, size: int = 100,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        assert not (size % 50), "Size not multiple of 50."
        self.save_hyperparameters()
        self._extra_out = {}
        self.xnorm = StaticNorm2D("sen2")
        self.ynorm = StaticNorm2D("sen3")
        self.gauss = AvgPool2d(3, 1, 1, count_include_pad=False)
        self.stem = FusionStem(13, 32, 12, size)
        self.match = ReScale2D()
        self.therm = ReflectiveToEmissive(6, 6)
        self.down_a = DownsamplingBlock( 32,  64, groups=2)
        self.down_b = DownsamplingBlock( 64, 128, groups=2)
        self.down_c = DownsamplingBlock(128, 256, groups=2)
        self.bridge = Bridge(256, 512) # 13x13
        self.up_c = FusionUpsamplingBlock(512, 256, size // 4)
        self.up_b = FusionUpsamplingBlock(256, 128, size // 2)
        self.up_a = FusionUpsamplingBlock(128,  64, size)
        self.head = Head(64, 12)
        self._initialize_weights()

        self.loss = msi2slstr_loss(a=2, b=1, c=1)
        self.strict_loading = True

    def _initialize_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, Conv2d) and m.weight.requires_grad:
                m.weight.data.normal_(std=sqrt(2) / sqrt(m.weight.shape[1]))
                if m.bias is not None:
                    m.bias.data.fill_(.0)

    def _thermal_training(self, x: Tensor, y: Tensor):
        """
        Train the thermal module and produce the high res thermal template.
        """
        optic_y = y[:, [0, 1, 2, 3, 4, 5]]

        # x scaled to value range of y.
        optic_x = self.match(x=x[:, DATA_CONFIG['sen2_bands']], ref=optic_y)

        self._extra_out['thermal_y'] = self.ynorm.denorm(
            self.therm(optic_y), channels=slice(6, 13, 1))

        # This serves only as target.
        # Eval not necessary as stats guaranteed to be same.
        with no_grad():
            # Average input for improved output.
            optic_x = self.gauss(optic_x)
            self._extra_out['thermal_x'] = self.ynorm.denorm(
                self.therm(optic_x).detach(), channels=slice(6, 13, 1))

    def forward(self, x, y) -> Tensor:
        x = self.xnorm(x)
        y = self.ynorm(y)
        a = self.stem(x, y)

        self._thermal_training(x, y)

        b = self.down_a(a)
        # self._extra_out['deep64'] = b

        c = self.down_b(b)
        # self._extra_out['deep128'] = c

        x = self.down_c(c)
        # self._extra_out['deep256'] = x

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

        loss, thermal = self.loss(x, Y_hat, y,
                                  self._extra_out['thermal_y'],
                                  self._extra_out['thermal_x'])

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

        loss, thermal = self.loss(x, Y_hat, y,
                                  self._extra_out['thermal_y'],
                                  self._extra_out['thermal_x'])

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

        loss, thermal = self.loss(x, Y_hat, y,
                                  self._extra_out['thermal_y'],
                                  self._extra_out['thermal_x'])

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
        return 0

    def configure_optimizers(self):
        opt = Adam(self.parameters(), lr=self.hparams.lr, maximize=True,
                    betas=(.9, .92), weight_decay=1e-10, eps=1e-6)
        sch = CosineAnnealingWarmRestarts(opt, T_0=100, T_mult=2,
                                          eta_min=1e-8)
        return [opt], [{"scheduler": sch, "interval": "step"}]


class msi2slstr_pretraining(msi2slstr):

    def __init__(self, lr: float = 0.001, *args: Any, **kwargs: Any) -> None:
        super().__init__(lr, *args, **kwargs)
        self.therm = emissivity_module.load_from_checkpoint(
            "pretrained/emissivity.ckpt").module

        self.loss = ssim(a=1, b=1, c=1)
        self.m2s = msi2slstr_loss()

    def training_step(self, batch, batch_idx) -> Tensor | Mapping[str, Any]:
        (x, y), _ = batch

        x = x.normal_(0, 1).mul(self.xnorm.var).add(self.xnorm.mean)

        Y_hat = self(x, y)

        self._extra_out['y'] = y
        self._extra_out['Y_hat'] = Y_hat

        # Main loss.
        loss = self.loss(self.m2s._usample(y), Y_hat)
        topo = self.loss.s(concat([x[:, [2, 3, 8, 10, 11, 12]],
                            self._extra_out["thermal_x"]], dim=1),
                            Y_hat)

        # For the OptToThermal optimization.
        thermal = self.loss(y[:, 6:], self._extra_out['thermal_y']).mean(0)

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

        return sum([batch_loss, deep_loss, thermal.mean(), topo.mean()])

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
    def __init__(self, lr: float = 0.001, size: int = 500,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(lr, size, *args, **kwargs)

    def predict_step(self, batch, batch_idx) -> None:
        return super().predict_step(batch, batch_idx)
    
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
