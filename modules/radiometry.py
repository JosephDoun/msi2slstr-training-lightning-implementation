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

from torch.nn import Conv2d
from torch.nn import AvgPool2d
from torch.nn import UpsamplingBilinear2d

from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader

from math import sqrt

from torch import rand
from torch import randn
from torch import randint
from torch import zeros
from torch import concat
from torch import Tensor

from .components import ReflectiveToEmissive
from .components import StaticNorm2D
from .components import Stem
from .components import ReScale2D
from .components import ChannelExpansion
from .components import ChannelCollapse
from .components import DownsamplingBlock
from .components import Bridge as Bridge
from .components import UpsamplingBlock
from .components import Head

from .emissivity import emissivity_module

from config import DATA_CONFIG
from metrics.ssim import cubic_ssim
from metrics.fusion import fusion_energy_metric
from metrics.fusion import fusion_topo_metric
from transformations.normalization import channel_stretch
from transformations.resampling import NonStrictAvgDownSamplingModule as Down


class radiometric_reconstruction_module(LightningModule):
    def __init__(self, lr: float = 1e-3, size: int = 100, strict: bool = True,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.strict_loading = strict
        self._extra_out = {}
        self.xnorm = StaticNorm2D("sen2")
        self.ynorm = StaticNorm2D("sen3")
        self.s24 = Stem(12, 24)
        self.d48 = DownsamplingBlock(24,  48)
        self.d96 = DownsamplingBlock(48,  96)
        self.d192 = DownsamplingBlock(96, 192)
        self.e180 = ChannelExpansion(-(-size // 8), 12, 180)
        self.c180 = ChannelCollapse(size // 50, 180, 12)
        self.b384 = Bridge(192, 384)
        self.u192 = UpsamplingBlock(384, 192, size // 4)
        self.u96 = UpsamplingBlock(192,  96, size // 2)
        self.u48 = UpsamplingBlock( 96,  48, size)
        self.h12 = Head(48, 12)

        # Helper modules.
        self._emissivity = emissivity_module.load_from_checkpoint(
            "pretrained/emissivity.ckpt").module
        self._gauss = AvgPool2d(3, 1, 1, count_include_pad=False)
        self._match = ReScale2D()

        self._initialize_weights()
        self._loss = cubic_ssim(agg='prod')
        self._fusion_energy_evaluation = fusion_energy_metric(agg='mean')
        self._fusion_topo_evaluation = fusion_topo_metric(agg='mean')
        self._up = UpsamplingBilinear2d((size, size))
        self._deep_up = UpsamplingBilinear2d((-(-size // 8), -(-size // 8)))

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

    def _training_scheme(self, x: Tensor):
        """
        Reconstruct low res image expected radiometry reinjection in latent
        space.
        """
        # Leveled input: Randomly scale input channels
        # to lose radiometric info.
        flat_in = self._mangle_radiometry(x)
        
        # Downsampled input as radiometric information..
        rad_in = Down(x)
        return x, flat_in, rad_in, rad_in

    def forward(self, x: Tensor, y: Tensor) -> Any:
        a = self.s24(x)
        b = self.d48(a)
        c = self.d96(b)
        x = self.d192(c)

        # Save reference to activation
        # for deep supervision.
        xy, xx = x.split_with_sizes((12, 180), dim=1)
        self._extra_out['a180'] = xx
 
        xx = self._match(xx, self.e180(y))    
        x = concat([xy.add(self._deep_up(y)), xx], dim=1)

        x = self.b384(x)
        x = self.u192(x, c)
        x = self.u96(x, b)
        x = self.u48(x, a)
        x = self.h12(x)
        return x

    def on_train_epoch_end(self) -> None:
        """
        Execute a sample prediction and log images.
        """
        # Run one extra sample during epoch end.
        loader: DataLoader = self.trainer.train_dataloader
        data, _ = loader.dataset[0]
        x, y = data
        x, y = (self.xnorm(x.cuda()), self.ynorm(y.cuda()))
        
        x = self._build_high_res_input(x)

        x, x_input, rad_in, _ = self._training_scheme(x)
        y = rad_in

        Y_hat = self(x_input, y)

        tboard: SummaryWriter = self.logger.experiment

        tboard.add_images(tag="training/x_in/train",
                          img_tensor=channel_stretch(x_input).swapaxes(0, 1),
                          global_step=self.current_epoch,
                          dataformats='NCHW')
        tboard.add_images(tag="training/x/train",
                          img_tensor=channel_stretch(x).swapaxes(0, 1),
                          global_step=self.current_epoch,
                          dataformats='NCHW')
        tboard.add_images(tag="training/Y_hat/train",
                          img_tensor=channel_stretch(Y_hat).swapaxes(0, 1),
                          global_step=self.current_epoch,
                          dataformats='NCHW')
    
    def training_step(self, batch, batch_idx) -> Tensor:
        """
        Generalized random training for radiometric recovery.
        """
        # Expects regular msi2slstr corregistered data.
        data, metadata = batch
        dates, tiles = metadata

        x, y = (self.xnorm(data[0]), self.ynorm(data[1]))

        # Thermal extrapolation helper.
        # Train the module on size 100x100 Y.
        thermal_y = self._emissivity(Down(x))
        thermal_loss = self._loss(thermal_y, y[:, 6:]).mean()

        # Get a) target, b) mangled input, c) radiometry and d) deep target.
        t_in, flat_in, rad_in, deep_in =\
        self._training_scheme(self._build_high_res_input(x))

        # Target prediction.
        Y_hat = self(flat_in, rad_in)

        # Estimated loss.
        loss = self._loss(t_in, Y_hat)

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
        
        # Deep loss should be precisely targeting the radiometry input.
        # .
        deep_loss = self._loss(
            # Additive collapse of activation to radiometry dimensions.
            self.c180(self._extra_out['a180']),
            # Downsampled raw input.
            Down(flat_in)
        ).mean()

        self.log("recon./deep/train", deep_loss,
                 logger=True, prog_bar=False, on_epoch=True,
                 on_step=True, batch_size=loss.size(0))

        return batch_loss.add(deep_loss).add(thermal_loss)
    
    def validation_step(self, batch, batch_idx) -> Tensor:
        """
        Validate reconstruction process.
        """
        batch, metadata = batch
        x, y = batch
        x, y = (self.xnorm(x), self.ynorm(y))

        # Roll over training workflows.
        # Get a) input, b) mangled input, c) radiometry and d) deep target.
        t_in, flat_in, rad_in, _ =\
        self._training_scheme(self._build_high_res_input(x))

        # Target prediction.
        Y_hat = self(flat_in, rad_in)
        
        # Estimated loss.
        loss = self._loss(t_in, Y_hat)
        
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
        x = self._build_high_res_input(x)
        # Target prediction.
        Y_hat = self(x, y)
        
        energy = self._fusion_energy_evaluation(y, Y_hat)
        topo = self._fusion_topo_evaluation(x, Y_hat)
        
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

        x = self._build_high_res_input(x)

        # Predict
        Y_hat = self.ynorm.denorm(self(x, y))

        # Write to output: A writer class should be
        # available to receive the inference product.
        self.trainer.predict_dataloaders.dataset.output(indices, Y_hat)
        return 0

    def configure_optimizers(self):
        opt = Adam(self.parameters(), lr=self.hparams.lr, maximize=True,
                    betas=(.9, .98), weight_decay=1e-10, eps=1e-6)
        sch1 = CosineAnnealingWarmRestarts(opt, T_0=100, T_mult=2,
                                           eta_min=1e-8)
        return [opt], [{"scheduler": sch1, "interval": "step"}]

