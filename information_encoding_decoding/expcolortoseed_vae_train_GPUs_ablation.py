import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader

from expcolortoseed_dataset_ablation import MyDataset
from models import UNet_Resblocks
from utils.config import (
    SPECIFIC_FOLDER_EXP_DIFFUSION_MORE,
    INVERSE_ABLATION_PHASE3_OUTPUT_ROOT,
    INVERSE_ABLATION_SEED32_SUBFOLDER,
)


ABLATION_CONFIGS = [
    ("grayscale", None),
    ("binary", None),
    ("gaussian", 1),
    ("gaussian", 2),
    ("gaussian", 4),
    ("gaussian", 5),
    ("bilateral", None),
    ("none", None),  # Unmodified RGB baseline; preserve existing task IDs 0-6.
]


def get_ablation_config():
    if "SLURM_ARRAY_TASK_ID" not in os.environ:
        raise RuntimeError(
            "SLURM_ARRAY_TASK_ID is not set. Submit this training with the "
            "ablation Slurm array script."
        )

    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if task_id < 0 or task_id >= len(ABLATION_CONFIGS):
        raise ValueError(
            f"SLURM_ARRAY_TASK_ID must be between 0 and "
            f"{len(ABLATION_CONFIGS) - 1}, received {task_id}."
        )

    return ABLATION_CONFIGS[task_id]


def main():
    ablation_type, gaussian_sigma = get_ablation_config()

    batch_size = 32
    learning_rate = 1e-4
    dropout = 0.3
    weight_decay = 1e-4
    lambda_ = 0.5
    pos_weight_value = 2.0
    pos_weight = torch.tensor([pos_weight_value])
    features = [16, 32, 64]

    ablation_name = ablation_type
    if ablation_type == "gaussian":
        ablation_name = f"gaussian_sigma{gaussian_sigma:g}"

    print(f"Running ablation: {ablation_name}")

    name = "Res_UNet_exps_color_vae_nonrandom_vDiffusion_ablation"
    version = (
        f"{ablation_name}_lambda{lambda_}_pos{pos_weight_value}_"
        f"features{features}_seeds_no_augmentation_crop3left2right_seed32_1gpu"
    )
    if os.environ.get('INVERSE_ABLATION_RETRY') == '1':
        version += f"_retry{os.environ.get('SLURM_ARRAY_JOB_ID', 'manual')}"

    model = UNet_Resblocks(
        in_channels=4,
        use_vae=True,
        features=features,
        learning_rate=learning_rate,
        lambda_=lambda_,
        pos_weight=pos_weight,
        dropout=dropout,
        weight_decay=weight_decay,
    )

    dataset = MyDataset(
        use_vae=True,
        source_folder=SPECIFIC_FOLDER_EXP_DIFFUSION_MORE,
        target_folder=os.path.join(INVERSE_ABLATION_PHASE3_OUTPUT_ROOT, 'N30000', INVERSE_ABLATION_SEED32_SUBFOLDER),
        preprocess_input="None",
        ablation_type=ablation_type,
        gaussian_sigma=gaussian_sigma,
        target_preprocessed=True,
    )

    n_total = 30000
    n_augmentations_per_image = 3
    n_val_base_images = 3000
    n_train = (n_total - n_val_base_images) * n_augmentations_per_image
    n_val = n_val_base_images * n_augmentations_per_image

    if len(dataset) < n_train + n_val:
        raise ValueError(
            f"Dataset has {len(dataset)} images, but the fixed split requires "
            f"{n_train + n_val}."
        )

    train_indices = list(range(0, n_train))
    val_indices = list(range(n_train, n_train + n_val))
    train_ds = torch.utils.data.Subset(dataset, train_indices)
    val_ds = torch.utils.data.Subset(dataset, val_indices)

    num_workers = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    train_loader = DataLoader(
        train_ds,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
        persistent_workers=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
        persistent_workers=True,
        pin_memory=True,
    )

    loggers = [
        TensorBoardLogger("lightning_logs", name=name, version=version),
        CSVLogger("lightning_logs", name=name, version=version),
    ]
    callbacks = [
        ModelCheckpoint(
            dirpath=f"lightning_logs/{name}/{version}/checkpoints",
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
        devices=1,
        precision=32,
        max_epochs=500,
        logger=loggers,
        callbacks=callbacks,
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
