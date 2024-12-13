"""
Experiment 2:

Substitution in image reconstruction.

Concept:
a. Train on reconstructing the radiometry and topography of coarse data.
b. Remove radiometry information from input and reinject in latent spaces.
c. Substitute input with high resolution but maintain multi-level injections.

"""
from typing import Any
from lightning import LightningModule

from torch import set_float32_matmul_precision
from torch import no_grad

from torch.nn import Conv2d
from torch.nn import AvgPool2d

from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard.writer import SummaryWriter

from torch import sqrt
from torch import randn
from torch import Tensor

from .components import OpticalToThermal
from .components import Stem
from .components import ReScale2D
from .components import ScaledProjection
from .components import DownsamplingBlock
from .components import Bridge
from .components import UpsamplingBlock
from .components import Head

from config import DATA_CONFIG
from metrics.ssim import cubic_ssim
from transformations.normalization import channel_stretch
from transformations.resampling import NonStrictAvgDownSamplingModule


class training(LightningModule):
    def __init__(self, size: int = 100, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._extra_out = {}
        self.stem = Stem(12, 16)
        self.down_a = DownsamplingBlock(16,  32)
        self.down_b = DownsamplingBlock(32,  64)
        self.down_c = DownsamplingBlock(64, 128)
        self.proj_a = ScaledProjection(12, 128, size // 4)
        self.bridge = Bridge(128, 256)
        self.up_c = UpsamplingBlock(256, 128, size // 4)
        self.up_b = UpsamplingBlock(128,  64, size // 2)
        self.up_a = UpsamplingBlock( 64,  32, size)
        self.head = Head(32, 12)
        self.match = ReScale2D()
        self.therm = OpticalToThermal(6, 6)
        self.gauss = AvgPool2d(3, 1, 1, count_include_pad=False)

        self._initialize_weights()
        self._loss = cubic_ssim()
        self._schemes = [self._training_scheme_1,
                         self._training_scheme_2,
                         self._training_scheme_3]

    def _initialize_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, Conv2d) and m.weight.requires_grad:
                m.weight.data.normal_(std=sqrt(2) / sqrt(m.weight.shape[1]))
                if m.bias is not None:
                    m.bias.data.fill_(.0)

    def _training_scheme_1(self, x: Tensor, y: Tensor):
        """
        Reconstruct low res image expected radiometry reinjection in latent
        space.
        """
        x_flat = x * randn((x.size(0), x.size(1), 1, 1))
        y = ...
        return self(x_flat, y)

    def _training_scheme_2(self, x: Tensor, y: Tensor):
        """
        Reconstruct high res image with latent injection of radiometry.

        Random channel selection.
        """
        return

    def _training_scheme_3(self, x: Tensor, y: Tensor):
        """
        Reconstruct high res image with latent injection of zeroes.

        Random channel selection.
        """
        return

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


    def forward(self, x: Tensor, y: Tensor) -> Any:
        x = self.xnorm(x)
        y = self.ynorm(y)
        a = self.stem(x, y)
        # Thermal extrapolation helper.
        self._thermal_training(x, y)
        b = self.down_a(a)
        # self._extra_out['a32'] = b
        c = self.down_b(b)
        # self._extra_out['a64'] = c
        x = self.down_c(c)
        # self._extra_out['a128'] = x
        x = self.bridge(x + self.proj_a(y))
        # The bridge activation should
        # target y directly.
        self._extra_out['a256'] = x
        x = self.up_c(x, c)
        x = self.up_b(x, b)
        x = self.up_a(x, a)
        x = self.head(x)
        return self.ynorm.denorm(x)

    def _log_experiment_metrics(self, *args, **kwargs):
        ...

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

    def training_step(self, batch, batch_idx) -> Tensor:
        """
        Generalized random training for radiometric recovery.
        """
        data, metadata = batch
        dates, tiles = metadata
        x, y = data
        x_flat = x * randn((x.size(0), x.size(1), 1, 1))
        Y_hat, loss = self._schemes[batch_idx % 3](x_flat, y)
        return loss
    
    def validation_step(self, batch, batch_idx) -> Tensor:
        """
        Validate reconstruction process.
        """
        return
    
    def test_step(self, batch, batch_idx) -> Tensor:
        """
        Mix inputs for fusion task. Should be evaluated using the fusion loss.
        """
        return

    def predict_step(self, batch, batch_idx) -> None:
        indices, (x, y) = batch

        # Predict
        Y_hat = self(x, y)

        # Write to output: A writer class should be
        # available to receive the inference product.
        self.trainer.predict_dataloaders.dataset.output(indices, Y_hat)
        return 0

    def configure_optimizers(self):
        opt = Adam(self.parameters(), lr=self.hparams.lr, maximize=True,
                    betas=(.9, .92), weight_decay=1e-10, eps=1e-6)
        sch = CosineAnnealingWarmRestarts(opt, T_0=100, T_mult=2,
                                          eta_min=1e-8)
        return [opt], [{"scheduler": sch, "interval": "step"}]
