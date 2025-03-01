"""
Experiment 2:

Substitution in image reconstruction.

Concept:
a. Train on reconstructing the radiometry and topography of coarse data.
b. Remove radiometry information from input and reinject in latent spaces.
c. Substitute input with high resolution but maintain multi-level injections.

"""
from typing import Any, Callable
from lightning import LightningModule
from lightning.pytorch.core.optimizer import LightningOptimizer

from random import choice

from torch.nn import Conv2d
from torch.nn import AvgPool2d
from torch.nn import UpsamplingBilinear2d
from torch.nn import UpsamplingNearest2d

from torch.optim import Adam

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import CyclicLR

from torchvision.transforms.functional import hflip
from torchvision.transforms.functional import vflip
from torchvision.transforms.functional import rotate

from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader

from math import sqrt

from torch import rand
from torch import randn
from torch import randn_like
from torch import randint
from torch import zeros
from torch import concat
from torch import arange
from torch import Tensor
from torch import no_grad

from .components import ReflectiveToEmissive
from .components import StaticNorm2D
from .components import Stem
from .components import DownsamplingBlock
from .components import VariationalBridge as Bridge
from .components import UpsamplingBlock
from .components import Head

from .emissivity import emissivity_module

from config import DATA_CONFIG
from metrics.ssim import ssim3d
from metrics.fusion import fusion_energy_metric
from metrics.fusion import fusion_topo_metric
from transformations.normalization import channel_stretch
from transformations.resampling import StrictAvgDownSamplingModule as Down
from transformations.resampling import ExpandedSpatialMetric as SpatialMetric


class radiometric_reconstruction_module(LightningModule):
    def __init__(self, lr: float = 1e-3, size: int = 300, strict: bool = True,
                 w_decay: float = 1e-5, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.strict_loading = strict
        self._extra_out = {}
        self.xnorm = StaticNorm2D("sen2")
        self.ynorm = StaticNorm2D("sen3")
        self.s16 = Stem([12, 13], 16)
        self.d32 = DownsamplingBlock(16,  32)
        self.d64 = DownsamplingBlock(32,  64)
        self.d128 = DownsamplingBlock(64, 128)
        self.b256 = Bridge(144, 256, -(-size // 8))
        self.u128 = UpsamplingBlock(256, 128, size // 4)
        self.u64 = UpsamplingBlock(128,  64, size // 2)
        self.u32 = UpsamplingBlock( 64,  32, size)
        self.h12 = Head(32, 12)

        # Helper modules.
        # TODO rename to template generator & move to top.
        self._emissivity = emissivity_module.load_from_checkpoint(
            "pretrained/emissivity.ckpt").module

        self._loss = ssim3d(agg='mean')
        self._fusion_energy_evaluation = fusion_energy_metric(agg='mean')
        self._fusion_topo_evaluation = fusion_topo_metric(agg='mean')
        self._to_top_size = UpsamplingBilinear2d((size, size))
        self._to_bottom_size = UpsamplingNearest2d((-(-size // 8),
                                                    -(-size // 8)))
        self._cell_avg = SpatialMetric("mean")
        self._cell_min = SpatialMetric("amin")
        self._cell_max = SpatialMetric("amax")

    def _initialize_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, Conv2d) and m.weight.requires_grad:
                m.weight.data.normal_(std=sqrt(2) / sqrt(m.weight.shape[1]))
                if m.bias is not None:
                    m.bias.data.fill_(.0)

    def _mangle_radiometry(self, x: Tensor) -> Tensor:
        """
        Randomly and independently scale, offset and add noise to input data.
        NOTE: Must not perform inplace operations on input tensor.
        """
        # Get intensity percentage map.
        distr = channel_stretch(x).square_()
        # Force expected value of 1.
        distr = distr.sub_(distr.mean((-1, -2), keepdim=True)).add_(1.)
        return x.clone().add_(
            # Low frequency noise.
            self._up(randn(x.size(0), x.size(1), 2, 2, device=x.device)
                     .add_(randn(x.size(0), x.size(1), 1, 1, device=x.device)))
                     .mul_(distr)
        )

    def _build_high_res_input(self, x: Tensor):
        """
        Constructs the expected input for the fusion task of Sentinel-2/3
        inputs.
        """
        # Use a filtered version of X to estimate emissivity.
        return concat([x[:, DATA_CONFIG["sen2_bands"]],
                       self._emissivity(self._gauss(x)).detach()], dim=-3)

    def _training_scheme(self, x: Tensor, y: Tensor):
        """
        Reconstruct low res image expected radiometry reinjection in latent
        space.
        """
        # Leveled input: Randomly scale input channels
        # to lose radiometric info.
        flat_in, alteration = self._mangle_radiometry(x)
        
        # Mean correction to be in line with real y.
        # Could be beneficial to learn realistic spectral signature tendencies.
        x = (x -
             x.mean((-1, -2), keepdim=True) +
             y.mean((-1, -2), keepdim=True))

        # Downsampled input as radiometric information to be injected.
        rad_in = Down(x)

        return (x,
                flat_in,
                rad_in,
                alteration)

    def forward(self, x: Tensor, y: Tensor) -> Any:
        a = self.s16(x)
        b = self.d32(a)
        c = self.d64(b)
        x = self.d128(c)
        x = self.b256(x, self._to_bottom_size(y))
        x = self.u128(x, c)
        x = self.u64(x, b)
        x = self.u32(x, a)
        x = self.h12(x)
        return x

    def on_train_epoch_end(self) -> None:
        """
        Execute a sample prediction and log images.
        """
        # Run one extra sample at epoch end.
        loader: DataLoader = self.trainer.train_dataloader
        data, _ = loader.dataset[0]
        x, y = data
        x, y = (self.xnorm(x.cuda()), self.ynorm(y.cuda()))
        
        template = self._build_high_res_target(x, y)

        template, h_input, radin, alteration = self._training_scheme(template, y)

        Y_hat = self([h_input, x], radin)

        tboard: SummaryWriter = self.logger.experiment

        tboard.add_images(tag="training/rad_in/train",
                          img_tensor=channel_stretch(radin).swapaxes(0, 1),
                          global_step=self.current_epoch,
                          dataformats='NCHW')
        tboard.add_images(tag="training/x/train",
                          img_tensor=channel_stretch(h_input).swapaxes(0, 1),
                          global_step=self.current_epoch,
                          dataformats='NCHW')
        tboard.add_images(tag="training/Y_hat/train",
                          img_tensor=channel_stretch(Y_hat).swapaxes(0, 1),
                          global_step=self.current_epoch,
                          dataformats='NCHW')
        tboard.add_images(tag="training/Y_hat/noise",
                          img_tensor=channel_stretch(alteration).swapaxes(0, 1),
                          global_step=self.current_epoch,
                          dataformats='NCHW')

    def _augment(self, x: Tensor, y: Tensor):
        x = x.add(randn_like(x).mul(.05))
        x = self._drop_band(x)

        if rand(1) > .5:
            x, y = hflip(x), hflip(y)

        if rand(1) > .5:
            x, y = vflip(x), vflip(y)

        return x, y

    def training_step(self, batch, batch_idx) -> Tensor:
        """
        Generalized random training for radiometric recovery.
        """
        # Expects regular msi2slstr corregistered data.
        data, metadata = batch
        # dates, tiles = metadata

        x, y = self.xnorm(data[0]), self.ynorm(data[1])

        x, y = self._augment(x, y)

        # Thermal extrapolation helper.
        # Trains over the low resolution input.
        # TODO rename.
        thermal_y = self._emissivity(Down(x))
        thermal_loss = self._loss(thermal_y, y).mean()

        # Goes over the high resolution input and create target Y.
        target = self._build_high_res_target(x, y)
        
        # Get a) pure input, b) mangled input, c) radiometry (downsampled pure input), d) noise introduced.
        target, h_input, rad_in, alteration = self._training_scheme(target, y)

        # Target prediction.
        Y_hat = self([h_input, x], rad_in)

        # Estimated loss.
        loss = self._loss(Y_hat, target)

        # Loss aggregation to build on.
        batch_loss = loss.mean()

        # For band evaluation.
        per_band = loss.mean(0)

        self.log("recon./train", batch_loss,
                 logger=True, prog_bar=True, on_epoch=True,
                 on_step=True, batch_size=loss.size(0))

        self.log_dict({**{f"recon./band_{i}/train": v for i, v in
                          enumerate(per_band)},
                       },
                      on_step=True,
                      on_epoch=True,
                      prog_bar=False,
                      logger=True,
                      batch_size=loss.size(0))

        # # Deep loss should be precisely targeting the downsampled model input.
        # deep_loss = self._loss(
        #     # Additive collapse of activation to radiometry dimensions.
        #     self.c192(self._extra_out['a192']),
        #     # Downsampled raw input.
        #     Down(h_input)
        # ).mean()

        # self.log("recon./deep/train", deep_loss,
        #          logger=True, prog_bar=False, on_epoch=True,
        #          on_step=True, batch_size=loss.size(0))

        return batch_loss.add(thermal_loss) #.add(deep_loss)

    def validation_step(self, batch, batch_idx) -> Tensor:
        """
        Validate reconstruction process.
        """
        batch, metadata = batch
        x, y = batch
        x, y = (self.xnorm(x), self.ynorm(y))

        target = self._build_high_res_target(x, y)

        # Get training inputs:
        # a) target, b) mangled input, c) radiometric injection, d) alteration tensor
        target, h_input, rad_in, _ = self._training_scheme(target, y)

        # Target prediction.
        Y_hat = self([h_input, x], rad_in)

        # Estimated loss.
        loss = self._loss(Y_hat, target)
        
        # Loss aggregation to build on.
        batch_loss = loss.mean()

        # For band evaluation.
        per_band = loss.mean(0)

        self.log("hp_metric", batch_loss, batch_size=loss.size(0))
        self.log("recon./val", batch_loss,
                 logger=True, prog_bar=True, on_epoch=True,
                 on_step=False, batch_size=loss.size(0))
        self.log_dict({**{f"recon./band_{i}/val": v for i, v in
                          enumerate(per_band)},
                       },
                      on_step=True,
                      on_epoch=True,
                      prog_bar=False,
                      logger=True,
                      batch_size=loss.size(0))

        return batch_loss
    
    def test_step(self, batch, batch_idx) -> Tensor:
        """
        Evaluation of the fusion results.
        """
        data, metadata = batch
        dates, tiles = metadata

        x, y = data
        x, y = (self.xnorm(x), self.ynorm(y))
        template = self._build_high_res_target(x, y)

        # Target prediction.
        Y_hat = self([template, x], y)

        energy = self._fusion_energy_evaluation(y, Y_hat)
        topo = self._fusion_topo_evaluation(template, Y_hat)

        self.log("hp_metric", energy.mean(), batch_size=energy.size(0))
        self.log("test/en", energy.mean(),
                 logger=True, prog_bar=True, on_epoch=True,
                 on_step=True, batch_size=energy.size(0))
        self.log("test/topo", topo.mean(),
                 logger=True, prog_bar=True, on_epoch=True,
                 on_step=True, batch_size=topo.size(0))
        self.log_dict({**{f"test/en/band_{i}": v for i, v in
                          enumerate(energy.mean(0))},
                       },
                      on_step=True,
                      on_epoch=True,
                      prog_bar=False,
                      logger=True,
                      batch_size=energy.size(0))
        self.log_dict({**{f"test/topo/band_{i}": v for i, v in
                          enumerate(topo.mean(0))},
                       },
                      on_step=True,
                      on_epoch=True,
                      prog_bar=False,
                      logger=True,
                      batch_size=topo.size(0))
        return energy.mul(topo).mean()

    def predict_step(self, batch, batch_idx) -> None:
        indices, (x, y) = batch
        x, y = self.xnorm(x), self.ynorm(y)

        template = self._build_high_res_target(x, y)

        # Predict
        Y_hat = self.ynorm.denorm(self([template, x], y))

        # Write to output: A writer class should be
        # available to receive the inference product.
        self.trainer.predict_dataloaders.dataset.output(indices, Y_hat)
        return 0

    def configure_optimizers(self):
        opt = Adam(self.parameters(), lr=self.hparams.lr, maximize=True,
                    betas=(.9, .999), weight_decay=self.hparams.w_decay,
                    eps=1e-8)

        sch = CosineAnnealingWarmRestarts(opt, 100, 2, 1e-6)
        sch = {"scheduler": sch, "interval": "step", "name": "lr/radiometry"}
        return [opt], [sch]
    
    def optimizer_step(self, epoch: int, batch_idx: int,
                       optimizer: Optimizer | LightningOptimizer,
                       optimizer_closure: Callable[[], Any] | None = None) -> None:
        
        # skip the first 1000 steps
        if self.trainer.global_step < 500:
            lr_scale = min(1.0, self.trainer.global_step >= 499 or 0)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr
        
        return super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
