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

from torch import no_grad

from torch.nn import Conv2d
from torch.nn import AvgPool2d

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
from .components import Bridge
from .components import UpsamplingBlock
from .components import Head

from .emissivity import emissivity_module

from config import DATA_CONFIG
from metrics.ssim import cubic_ssim
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
        self.e192 = ChannelExpansion(-(-size // 8), 12, 192)
        self.b384 = Bridge(192, 384)
        self.c384 = ChannelCollapse(size // 50, 384, 12)
        self.u192 = UpsamplingBlock(384, 192, size // 4)
        self.u96 = UpsamplingBlock(192,  96, size // 2)
        self.u48 = UpsamplingBlock( 96,  48, size)
        self.h12 = Head(48, 12)

        # Helper modules for training.
        self.match = ReScale2D()
        self._emissivity = emissivity_module.load_from_checkpoint(
            "pretrained/emissivity.ckpt").module
        self._gauss = AvgPool2d(3, 1, 1, count_include_pad=False)

        self._initialize_weights()
        self._loss = cubic_ssim(agg='prod')
        self._schemes = [self._training_scheme_1,
                         self._training_scheme_2]

    def _initialize_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, Conv2d) and m.weight.requires_grad:
                m.weight.data.normal_(std=sqrt(2) / sqrt(m.weight.shape[1]))
                if m.bias is not None:
                    m.bias.data.fill_(.0)

    def _mangle_radiometry(self, x: Tensor):
        """
        Randomly and independently scale and offset channel values.
        """
        # 0.5, 1.5
        scale = rand(x.size(0), x.size(1), 1, 1, device=x.get_device())\
            .mul(1.).add(.5)
        # +- .5
        offset = randn(x.size(0), x.size(1), 1, 1, device=x.get_device())\
            .mul(.5)
        return x.mul(scale).add(offset).add(
            # Added noise.
            randn(*x.shape, device=x.get_device()).mul_(.1)
        )
    
    def _dropout(self, x: Tensor):
        """
        Dropout a random input band.
        """
        ...

    def _build_high_res_input(self, batch):
        """
        Constructs the expected input for the fusion task of Sentinel-2/3
        inputs.
        """
        x, y = batch
        x_o = x[:, DATA_CONFIG["sen2_bands"]]
        # Use a filtered version of X to estimate emissivity.
        x_t = self._emissivity(self._gauss(x_o)).detach()
        return concat([x_o, x_t], dim=-3)

    def get_random_training_input(self, batch):
            """
            Randomly select between first and second batch elements.
            """
            return self._build_high_res_input(batch) if rand(1) > .5 else\
                batch[1]

    def _training_scheme_1(self, batch):
        """
        Reconstruct low res image expected radiometry reinjection in latent
        space.
        """
        # Input tensor.
        t_in = self.get_random_training_input(batch)
        
        # Leveled input: Randomly scale input channels
        # to lose radiometric info.
        flat_in = self._mangle_radiometry(t_in)
        
        # Downsampled input as radiometric information..
        rad_in = Down(t_in)
        return t_in, flat_in, rad_in, rad_in

    def _training_scheme_2(self, batch):
        """
        The zero case. Reconstruct image when latent injection is zero,
        expecting no effect.

        # NOTE: Random channel selection?

        All targets are represented as `flat_in`, as `rad_in` does not
        contribute. 
        """
        t_in = self.get_random_training_input(batch)
        flat_in = self._mangle_radiometry(t_in)
        rad_in = Down(flat_in).fill_(0)
        # Target is the tampered radiometry input.
        return flat_in, flat_in, rad_in, Down(flat_in)

    def forward(self, x: Tensor, y: Tensor) -> Any:
        # x = self.xnorm(x)
        # y = self.ynorm(y)
        a = self.s24(x)
        b = self.d48(a)
        c = self.d96(b)
        x = self.d192(c)
        x = self.b384(x + self.e192(y))
        # Activation save for deep supervision.
        self._extra_out['a384'] = x
        x = self.u192(x, c)
        x = self.u96(x, b)
        x = self.u48(x, a)
        x = self.h12(x)
        return x # self.ynorm.denorm(x)

    def on_train_epoch_end(self) -> None:
        """
        Execute a sample prediction and log images.
        """
        # Run one extra sample during epoch end.
        loader: DataLoader = self.trainer.train_dataloader
        x, y = loader.dataset[0]
        x, y = (self.xnorm(x.cuda()), self.ynorm(y.cuda()))
        
        x = self._build_high_res_input((x, y))
        
        y = Down(x).fill_(0)
        
        Y_hat = self(x, y)

        tboard: SummaryWriter = self.logger.experiment

        tboard.add_images(tag="training/x/train",
                          img_tensor=channel_stretch(x).swapaxes(0, 1),
                          global_step=self.current_epoch,
                          dataformats='NCHW')

        tboard.add_images(tag="training/Y_hat/train",
                          img_tensor=channel_stretch(Y_hat).swapaxes(0, 1),
                          global_step=self.current_epoch,
                          dataformats='NCHW')
    
    def on_train_start(self) -> None:
        super().on_train_start()
        self._emissivity.eval()
    
    def training_step(self, batch, batch_idx) -> Tensor:
        """
        Generalized random training for radiometric recovery.
        """
        batch = (self.xnorm(batch[0]), self.ynorm(batch[1]))
        # x, y = batch

        # Thermal extrapolation helper.
        # Train the module on size 100x100 Y.
        # thermal_y = self._thermal_training(x, y)
        # thermal_loss = self._loss(thermal_y, y[:, 6:]).mean()

        # Roll over training workflows.
        # Get a) target, b) mangled input, c) radiometry and d) deep target.
        t_in, flat_in, rad_in, deep_in =\
        self._schemes[batch_idx % len(self._schemes)](batch)

        # Target prediction.
        Y_hat = self(flat_in, rad_in)
        
        # Estimated loss.
        loss = self._loss(t_in, Y_hat)
        
        # Loss aggregation to build on.
        batch_loss = loss.mean()

        # For band evaluation.
        per_band = loss.mean(0)

        self.log("hp_metric", batch_loss)
        self.log("training/loss/train", batch_loss,
                 logger=True, prog_bar=True, on_epoch=True,
                 on_step=True, batch_size=per_band.size(0))
        self.log_dict({**{f"training/band_{i}/train": v for i, v in
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
            self.c384(self._extra_out['a384']),
            # Downsampled input.
            deep_in
        ).mean()

        self.log("training/deep_supervision/train", deep_loss,
                 logger=True, prog_bar=False, on_epoch=True,
                 on_step=True, batch_size=per_band.size(0))

        return batch_loss.add(deep_loss) # .add(thermal_loss)
    
    def validation_step(self, batch, batch_idx) -> Tensor:
        """
        Validate reconstruction process.
        """
        batch = (self.xnorm(batch[0]), self.ynorm(batch[1]))

        # Roll over training workflows.
        # Get a) input, b) mangled input, c) radiometry and d) deep target.
        t_in, flat_in, rad_in, _ =\
        self._schemes[batch_idx % len(self._schemes)](batch)

        # Target prediction.
        Y_hat = self(flat_in, rad_in)
        
        # Estimated loss.
        loss = self._loss(t_in, Y_hat)
        
        # Loss aggregation to build on.
        batch_loss = loss.mean()

        # For band evaluation.
        per_band = loss.mean(0)

        self.log("hp_metric", batch_loss)
        self.log("training/loss/valid", batch_loss,
                 logger=True, prog_bar=True, on_epoch=True,
                 on_step=True, batch_size=per_band.size(0))
        self.log_dict({**{f"training/band_{i}/valid": v for i, v in
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
        Mix inputs for fusion task fed by the msi2slstr dataset.
        Should be evaluated using the fusion loss.
        TODO
        """
        data, metadata = batch
        dates, tiles = metadata
        batch = (self.xnorm(batch[0]), self.ynorm(batch[1]))
        x, y = batch
        x = self._build_high_res_input(batch=batch)
        # Target prediction.
        Y_hat = self(x, y)
        
        # Estimated loss. NOTE: Needs fusion loss.
        # TODO
        loss = ... # self._loss(x, y, Y_hat)
        
        # Loss aggregation to build on.
        batch_loss = loss.mean()

        # For band evaluation.
        per_band = loss.mean(0)

        self.log("hp_metric", batch_loss)
        self.log("training/loss/test", batch_loss,
                 logger=True, prog_bar=True, on_epoch=True,
                 on_step=True, batch_size=per_band.size(0))
        self.log_dict({**{f"training/band_{i}/test": v for i, v in
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
            self.c384(self._extra_out['a384']),
            # Downsampled input.
            y
        ).mean()

        self.log("training/deep_supervision/test", deep_loss,
                 logger=True, prog_bar=False, on_epoch=True,
                 on_step=True, batch_size=per_band.size(0))
        return batch_loss

    def predict_step(self, batch, batch_idx) -> None:
        indices, (x, y) = batch
        x, y = self.xnorm(x), self.ynorm(y)

        x = self._build_high_res_input((x, y))

        # Predict
        Y_hat = self.ynorm.denorm(self(x, y))

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


class latent_substitution_experiment(radiometric_reconstruction_module):
    """
    Force the bridge to target the coarse resolution version during an
    auto-encoding task, without injection.

    This would make then the latent space substitutable with other
    radiometric sources.

    TODO
    """
    def __init__(self, size: int = 100, *args: Any, **kwargs: Any) -> None:
        super().__init__(size, *args, **kwargs)

    def training_step(self, batch, batch_idx) -> Tensor:
        ...
