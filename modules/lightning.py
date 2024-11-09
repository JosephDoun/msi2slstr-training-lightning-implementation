from lightning import LightningModule

from metrics.fusion import msi2slstr_loss
from metrics.ssim import ssim
from metrics.supervision import DeepLoss

from transformations.normalization import channel_stretch

from typing import Dict, Mapping
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

from torch.nn import CosineSimilarity
from torch.nn import L1Loss

set_float32_matmul_precision('medium')

MSI2SLSTRLoss = msi2slstr_loss()
SSIM = ssim()
DEEPLOSS = DeepLoss([64, 128, 256, 512], [50,  25,  13,  13],
                    loss_fn=CosineSimilarity(dim=2),
                    maximize=True)


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
            if isinstance(m, Conv2d) and m.weight.requires_grad:
                m.weight.data.normal_(std=sqrt(2) / sqrt(m.weight.shape[1]))
                if m.bias is not None:
                    m.bias.data.fill_(.0)

    def forward(self, x, y) -> Tensor:
        optic_y = y[:, [0, 1, 2, 3, 4, 5]]
        optic_x = self.scale(x[:, [2, 3, 8, 10, 11, 12]], optic_y)

        # Only needed during training.
        self._extra_out['thermal_y'] = self.therm(optic_y)
        
        # This serves only as target.
        self.therm.eval()
        self._extra_out['thermal_x'] = self.therm(optic_x).detach()

        a = self.stem(x, y)

        b = self.down_a(a)
        self._extra_out['deep64'] = b

        c = self.down_b(b)
        self._extra_out['deep128'] = c

        x = self.down_c(c)
        self._extra_out['deep256'] = x

        x = self.bridge(x)
        self._extra_out['deep512'] = x

        x = self.up_c(x, c)
        x = self.up_b(x, b)
        x = self.up_a(x, a)
        return self.head(x)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        return super().on_load_checkpoint(checkpoint)

    def training_step(self, batch, batch_idx) -> Tensor | Mapping[str, Any]:
        data, metadata = batch
        dates, tiles = metadata
        x, y = data

        Y_hat = self(x, y)
        loss = MSI2SLSTRLoss(x, Y_hat, y, self._extra_out['thermal_y'],
                 self._extra_out['thermal_x'])
        
        # For channelwise evaluation.
        energy = MSI2SLSTRLoss.energy(y, Y_hat).mean(0)
        thermal = MSI2SLSTRLoss.evaluate(y[:, 6:],
                                         self._extra_out['thermal_y']).mean(0)

        self._extra_out['x'] = x
        self._extra_out['y'] = y
        self._extra_out['Y_hat'] = Y_hat

        batch_loss = loss.mean()
        # For samplewise evaluation.
        sample_loss = loss.detach().mean(-1)

        self.log("hp_metric", batch_loss)
        self.log("loss/train", batch_loss,
                 logger=True, prog_bar=True, on_epoch=True,
                 on_step=True, batch_size=sample_loss.size(0))

        self.log_dict({**{f"energy_{i}/train": v for i, v in
                          enumerate(energy)},

                       **{f"thermal_{i}/train": v for i, v in
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

        deep_loss = DEEPLOSS(self._extra_out['deep64'],
                             self._extra_out['deep128'],
                             self._extra_out['deep256'],
                             self._extra_out['deep512'],
                             y=y).mean()

        return batch_loss + deep_loss
    
    def on_train_epoch_end(self) -> None:
        tboard: SummaryWriter = self.logger.experiment

        tboard.add_images(tag="x/train",
                          img_tensor=channel_stretch(self._extra_out['x'][1])
                                                     .unsqueeze(1),
                          global_step=self.current_epoch,
                          dataformats='NCHW')

        tboard.add_images(tag="y/train",
                          img_tensor=channel_stretch(self._extra_out['y'][1])
                                                     .unsqueeze(1),
                          global_step=self.current_epoch,
                          dataformats='NCHW')

        tboard.add_images(tag="Y_hat/train",
                          img_tensor=channel_stretch(self._extra_out['Y_hat']
                                                     [1]).unsqueeze(1),
                          global_step=self.current_epoch,
                          dataformats='NCHW')

    def validation_step(self, batch, batch_idx) -> Tensor | Mapping[str, Any]:
        data, metadata = batch
        dates, tiles = metadata
        x, y = data

        Y_hat = self(x, y)
        loss = MSI2SLSTRLoss(x, Y_hat, y, self._extra_out['thermal_y'],
                 self._extra_out['thermal_x'])

        # For channelwise evaluation.
        energy = MSI2SLSTRLoss.energy(y, Y_hat).mean(0)
        thermal = MSI2SLSTRLoss.evaluate(y[:, 6:],
                                         self._extra_out['thermal_y']).mean(0)

        batch_loss = loss.mean()
        sample_loss = loss.detach().mean(-1)

        self.log("loss/val", batch_loss, prog_bar=True, on_epoch=True,
                 batch_size=sample_loss.size(0))

        self.log_dict({**{f"energy_{i}/val": v for i, v in
                          enumerate(energy)},

                       **{f"thermal_{i}/val": v
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
        
        loss, thermal = MSI2SLSTRLoss(x, Y_hat, y,
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

    def configure_optimizers(self):
        opt = Adam(self.parameters(), lr=self.hparams.lr, maximize=True,
                    betas=(.9, .92))
        sch = CosineAnnealingWarmRestarts(opt, T_0=100, T_mult=2,
                                          eta_min=1e-8)
        return [opt], [{"scheduler": sch, "interval": "step"}]


class msi2slstr_pretraining(msi2slstr):
    def training_step(self, batch, batch_idx) -> Tensor | Mapping[str, Any]:
        (x, y), _ = batch
        
        x.normal_(0, .01)
        Y_hat = self(x, y)
        
        self._extra_out['y'] = y
        self._extra_out['Y_hat'] = Y_hat

        # Main loss.
        loss = SSIM(MSI2SLSTRLoss._usample(y), Y_hat)

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

        return batch_loss + deep_loss + thermal.mean()

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
        ...

    def test_step(self, batch, batch_idx) -> Tensor | Mapping[str, Any]:
        ...


class thermal_prediction(LightningModule):
    def __init__(self, lr=1e-3, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.module = OpticalToThermal(6, 6)
        self.save_hyperparameters()

    def forward(self, x):
        return self.module(x)

    def training_step(self, batch, batch_idx) -> Tensor | Mapping[str, Any]:
        Y_hat = self(batch[:, :6])
        loss = SSIM(batch[:, 6:], Y_hat)
        batch_loss = loss.mean()
        eval_loss = SSIM.evaluate(batch[:, 6:], Y_hat)
        sample_loss = eval_loss.mean(0)
        
        self.log("hp_metric", eval_loss.mean(), batch_size=sample_loss.size(0))
        self.log("opt_to_thermal_experiment/train/loss",
                 eval_loss.mean(),
                 batch_size=batch.size(0), on_step=True, prog_bar=True)
        
        self.log_dict({f"opt_to_thermal_experiment/train_{i}": v for i, v in
                       enumerate(sample_loss)}, batch_size=batch.size(0),
                       on_step=True)

        return batch_loss

    def validation_step(self, batch, batch_idx) -> Tensor | Mapping[str, Any]:
        Y_hat = self(batch[:, :6])
        loss = SSIM(batch[:, 6:], Y_hat)
        batch_loss = loss.mean()
        eval_loss = SSIM.evaluate(batch[:, 6:], Y_hat)
        sample_loss = eval_loss.mean(0)
        
        self.log("opt_to_thermal_experiment/val/loss",
                 eval_loss.mean(),
                 batch_size=batch.size(0), on_step=True, prog_bar=True)
        
        self.log_dict({f"opt_to_thermal_experiment/val_{i}": v for i, v in
                       enumerate(sample_loss)}, batch_size=batch.size(0),
                       on_step=True)
        return batch_loss

    def test_step(self, batch, batch_idx) -> Tensor | Mapping[str, Any]:
        Y_hat = self(batch[:, :6])
        loss = SSIM(batch[:, 6:], Y_hat)
        batch_loss = loss.mean()
        eval_loss = SSIM.evaluate(batch[:, 6:], Y_hat)
        sample_loss = eval_loss.mean(0)

        self.log("opt_to_thermal_experiment/test/loss",
                 eval_loss.mean(),
                 batch_size=batch.size(0), on_step=True, prog_bar=True)

        self.log_dict({f"opt_to_thermal_experiment/test_{i}": v for i, v in
                       enumerate(sample_loss)}, batch_size=batch.size(0),
                       on_step=True)

        return batch_loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr, maximize=True)
