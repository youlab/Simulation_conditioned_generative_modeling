import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader

from colonyarea_dataset import MyDataset
from models.ResNet import ColonyMetricsResNet
from utils.config import SPECIFIC_FOLDER_EXP_DIFFUSION_MORE


# Training configuration
batch_size = 32
learning_rate = 1e-4
dropout = 0.3
weight_decay = 1e-4
features = [16, 32, 64]
hidden_dim = 64

# The JSON file is saved one directory above the image folder.
metrics_json_path = os.path.join(
    os.path.dirname(os.path.abspath(SPECIFIC_FOLDER_EXP_DIFFUSION_MORE)),
    "colony_metrics.json",
)

dataset = MyDataset(
    source_folder=SPECIFIC_FOLDER_EXP_DIFFUSION_MORE,
    metrics_json_path=metrics_json_path,
    preprocess_input="None",
)


# Keep all three replicates of each base image in the same split.
# Natural filename sorting places the three replicates next to each other.
n_replicates_per_image = 3
n_val_base_images = 3000

if len(dataset) % n_replicates_per_image != 0:
    raise ValueError(
        f"Dataset size {len(dataset)} is not divisible by "
        f"{n_replicates_per_image} replicates per image"
    )

n_total_base_images = len(dataset) // n_replicates_per_image
if n_val_base_images >= n_total_base_images:
    raise ValueError("The validation split must contain fewer images than the full dataset")

n_train = (n_total_base_images - n_val_base_images) * n_replicates_per_image
n_val = n_val_base_images * n_replicates_per_image

train_indices = list(range(0, n_train))
val_indices = list(range(n_train, n_train + n_val))

train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)


num_gpus = 4
num_workers = max(1, (os.cpu_count() or 1) // num_gpus)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    persistent_workers=True,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    persistent_workers=True,
    pin_memory=True,
)


model = ColonyMetricsResNet(
    use_vae=True,
    features=features,
    hidden_dim=hidden_dim,
    dropout=dropout,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    tip_scale=64.0,
    log_transform_tips=False,
)

name = "ResNet_exps_colony_metrics_vae"
version = f"MSE_features{features}_no_tip_log_transform"

loggers = [
    TensorBoardLogger("lightning_logs", name=name, version=version),
    CSVLogger("lightning_logs", name=name, version=version),
]

checkpoint_directory = f"lightning_logs/{name}/{version}/checkpoints"
callbacks = [
    ModelCheckpoint(
        dirpath=checkpoint_directory,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="best-{epoch:03d}-{val_loss:.4f}",
    ),
    EarlyStopping(monitor="val_loss", mode="min", patience=50),
    LearningRateMonitor(logging_interval="epoch"),
]


trainer = pl.Trainer(
    accelerator="gpu",
    devices=num_gpus,
    strategy="ddp",
    precision=32,
    max_epochs=500,
    logger=loggers,
    callbacks=callbacks,
)

trainer.fit(model, train_loader, val_loader)
