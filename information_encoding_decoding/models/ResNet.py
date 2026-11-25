"""VAE–ResNet colony-area inverse model (Figure 5G).

The trained checkpoint also has a tip-count head; it is retained for checkpoint
compatibility. Paper evaluation plots use the colony-area output only.
"""
"""Residual multi-task regressor for colony morphology metrics.

The model predicts two image-level quantities:
    1. colony area fraction
    2. number of boundary tips

It can operate on RGB images directly or on frozen Stable Diffusion VAE latents,
matching the input representation used by the earlier inverse-model pipeline.
"""

from typing import Dict, Sequence, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Two-convolution residual block with an optional channel projection."""

    def __init__(self, channels_in: int, channels_out: int, dropout: float = 0.0):
        super().__init__()
        self.projection = (
            nn.Conv2d(channels_in, channels_out, kernel_size=1)
            if channels_in != channels_out
            else nn.Identity()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels_out),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.projection(x)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return self.activation(x + identity)


class ColonyMetricsResNet(pl.LightningModule):
    """Predict colony area fraction and tip count from an image.

    Expected batch keys are ``source``, ``colony_area_fraction``, and
    ``number_of_tips``. Alternatively, a ``target`` tensor of shape ``(B, 2)``
    can be supplied, with area fraction in column 0 and tip count in column 1.

    When ``use_vae=True``, input images must be RGB tensors normalized to [0, 1].
    The frozen VAE converts them to four-channel latent tensors before the
    residual encoder. VAE posterior means are used so validation and inference
    remain deterministic.
    """

    def __init__(
        self,
        use_vae: bool = True,
        in_channels: int | None = None,
        features: Sequence[int] = (32, 64, 128),
        hidden_dim: int = 64,
        dropout: float = 0.2,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        area_loss_weight: float = 1.0,
        tip_loss_weight: float = 1.0,
        tip_scale: float = 64.0,
        log_transform_tips: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        if not features:
            raise ValueError("features must contain at least one channel size")
        if tip_scale <= 0:
            raise ValueError("tip_scale must be positive")

        self.use_vae = use_vae
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.area_loss_weight = area_loss_weight
        self.tip_loss_weight = tip_loss_weight
        self.tip_scale = float(tip_scale)
        self.log_transform_tips = log_transform_tips

        if use_vae:
            # Import lazily so a direct-RGB model does not load the large VAE.
            from models.vae import vae

            self.vae = vae
            self.vae.eval()
            self.vae.requires_grad_(False)
            encoder_channels = 4
        else:
            self.vae = None
            encoder_channels = 3 if in_channels is None else in_channels

        self.encoder = nn.ModuleList()
        for feature in features:
            self.encoder.append(ResBlock(encoder_channels, feature, dropout=dropout))
            encoder_channels = feature

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.shared_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(features[-1], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.area_head = nn.Linear(hidden_dim, 1)
        self.tip_head = nn.Linear(hidden_dim, 1)

    def train(self, mode: bool = True):
        """Keep the frozen VAE in evaluation mode during model training."""
        super().train(mode)
        if self.vae is not None:
            self.vae.eval()
        return self

    def encode_source(self, x: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            return x

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if x.shape[1] != 3:
            raise ValueError(f"The VAE expects 1 or 3 image channels, received {x.shape[1]}")

        with torch.no_grad():
            posterior = self.vae.encode(x * 2.0 - 1.0).latent_dist
            # The earlier pipeline sampled here. The posterior mean is preferable
            # for deterministic scalar regression and reproducible validation.
            return 0.18215 * posterior.mean

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encode_source(x)
        for block in self.encoder:
            x = block(x)
            x = self.pool(x)

        features = self.shared_head(self.global_pool(x))
        area_fraction = torch.sigmoid(self.area_head(features)).squeeze(-1)
        transformed_tips = self.tip_head(features).squeeze(-1)
        return area_fraction, transformed_tips

    def transform_tip_target(self, tips: torch.Tensor) -> torch.Tensor:
        tips = tips.float()
        if self.log_transform_tips:
            return torch.log1p(tips)
        # Scaling preserves ordinary count-space MSE up to a constant factor,
        # while keeping its magnitude comparable to area-fraction MSE.
        return tips / self.tip_scale

    def inverse_tip_prediction(self, transformed_tips: torch.Tensor) -> torch.Tensor:
        if self.log_transform_tips:
            tips = torch.expm1(transformed_tips)
        else:
            tips = transformed_tips * self.tip_scale
        return torch.clamp(tips, min=0.0)

    def _tensorboard_experiment(self):
        """Return a TensorBoard SummaryWriter when one is among the loggers."""
        trainer = getattr(self, "_trainer", None)
        loggers = trainer.loggers if trainer is not None else []
        for logger in loggers:
            experiment = getattr(logger, "experiment", None)
            if experiment is not None and hasattr(experiment, "add_image"):
                return experiment
        return None

    def _log_prediction_examples(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        stage: str,
        area_target: torch.Tensor,
        tip_target: torch.Tensor,
        area_prediction: torch.Tensor,
        tip_prediction: torch.Tensor,
    ) -> None:
        """Log sparse visual and numerical prediction checks to TensorBoard."""
        if stage == "train":
            batches_to_log = {0}
        elif stage == "val":
            batches_to_log = {0, 10, 20, 30}
        else:
            return
        if batch_idx not in batches_to_log or batch["source"].shape[0] == 0:
            return

        experiment = self._tensorboard_experiment()
        if experiment is None:
            return

        sample_index = 0
        example_index = sorted(batches_to_log).index(batch_idx)
        tag_prefix = f"{stage}/example_{example_index}"
        source = batch["source"][sample_index].detach().float().cpu()
        experiment.add_image(f"{tag_prefix}/source", source, self.current_epoch)

        if self.use_vae:
            with torch.no_grad():
                latent = self.encode_source(batch["source"][sample_index:sample_index + 1])
            latent_image = latent[0].mean(dim=0, keepdim=True).detach().float().cpu()
            latent_min, latent_max = latent_image.min(), latent_image.max()
            latent_image = (latent_image - latent_min) / (latent_max - latent_min + 1e-8)
            experiment.add_image(f"{tag_prefix}/vae_latent", latent_image, self.current_epoch)

        area_true = area_target[sample_index].item()
        area_pred = area_prediction[sample_index].item()
        tips_true = tip_target[sample_index].item()
        tips_pred = tip_prediction[sample_index].item()
        comparison = (
            "| Metric | Target | Prediction | Absolute error |\n"
            "|---|---:|---:|---:|\n"
            f"| Colony area fraction | {area_true:.6f} | {area_pred:.6f} | "
            f"{abs(area_pred - area_true):.6f} |\n"
            f"| Number of tips | {tips_true:.2f} | {tips_pred:.2f} | "
            f"{abs(tips_pred - tips_true):.2f} |"
        )
        experiment.add_text(f"{tag_prefix}/target_vs_prediction", comparison, self.current_epoch)

    @staticmethod
    def unpack_targets(batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if "target" in batch:
            target = batch["target"]
            if target.ndim != 2 or target.shape[1] != 2:
                raise ValueError("batch['target'] must have shape (batch_size, 2)")
            return target[:, 0].float(), target[:, 1].float()

        required = {"colony_area_fraction", "number_of_tips"}
        missing = required.difference(batch)
        if missing:
            raise KeyError(f"Batch is missing target key(s): {sorted(missing)}")
        return batch["colony_area_fraction"].float(), batch["number_of_tips"].float()

    def calculate_loss(
        self,
        area_prediction: torch.Tensor,
        transformed_tip_prediction: torch.Tensor,
        area_target: torch.Tensor,
        tip_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        transformed_tip_target = self.transform_tip_target(tip_target)

        # Active baseline: ordinary mean-squared error for both targets.
        area_loss = F.mse_loss(area_prediction, area_target)
        tip_loss = F.mse_loss(transformed_tip_prediction, transformed_tip_target)

        # Smooth-L1 alternatives for a later robustness experiment:
        # area_loss = F.smooth_l1_loss(area_prediction, area_target)
        # tip_loss = F.smooth_l1_loss(transformed_tip_prediction, transformed_tip_target)

        total_loss = (
            self.area_loss_weight * area_loss
            + self.tip_loss_weight * tip_loss
        )
        return total_loss, area_loss, tip_loss

    def _shared_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        stage: str,
    ) -> torch.Tensor:
        area_target, tip_target = self.unpack_targets(batch)
        area_prediction, transformed_tip_prediction = self(batch["source"])
        loss, area_loss, tip_loss = self.calculate_loss(
            area_prediction,
            transformed_tip_prediction,
            area_target,
            tip_target,
        )
        tip_prediction = self.inverse_tip_prediction(transformed_tip_prediction)

        self._log_prediction_examples(
            batch=batch,
            batch_idx=batch_idx,
            stage=stage,
            area_target=area_target,
            tip_target=tip_target,
            area_prediction=area_prediction,
            tip_prediction=tip_prediction,
        )

        batch_size = batch["source"].shape[0]
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True,
                 on_step=stage == "train", batch_size=batch_size, sync_dist=True)
        self.log(f"{stage}_area_mse", area_loss, on_epoch=True,
                 on_step=False, batch_size=batch_size, sync_dist=True)
        self.log(f"{stage}_tip_loss", tip_loss, on_epoch=True,
                 on_step=False, batch_size=batch_size, sync_dist=True)
        self.log(f"{stage}_area_mae", F.l1_loss(area_prediction, area_target),
                 on_epoch=True, on_step=False, batch_size=batch_size, sync_dist=True)
        self.log(f"{stage}_tip_mae", F.l1_loss(tip_prediction, tip_target),
                 on_epoch=True, on_step=False, batch_size=batch_size, sync_dist=True)
        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, "test")

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        area_fraction, transformed_tips = self(batch["source"])
        return {
            "colony_area_fraction": area_fraction,
            "number_of_tips": self.inverse_tip_prediction(transformed_tips),
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            },
        }
