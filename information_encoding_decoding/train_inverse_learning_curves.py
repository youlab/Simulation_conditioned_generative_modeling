"""Train one learning-curve condition per Slurm task from explicit paired CSVs."""
import csv
import json
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader

from expcolortoseed_dataset import MyDataset
from models import UNet_Resblocks
from utils import config


class ManifestDataset(MyDataset):
    """Load explicit pairs, then reuse MyDataset's existing image preprocessing."""

    def __init__(self, manifest_path):
        with open(manifest_path, newline='') as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f'Empty manifest: {manifest_path}')

        self.source_files = [row['source_path'] for row in rows]
        self.target_files = [row['target_path'] for row in rows]
        self.configuration_ids = {row['configuration_id'] for row in rows}

        # Phase 2/3 already cropped these images. Do not crop them again.
        self.use_vae = True
        self.preprocess_input = 'preprocessed'
        self.target_preprocessed = True  # Read native 32x32 labels without resampling.


def selected_condition():
    # Direct invocation trains task 0 only. Use the launcher for all conditions.
    index = int(os.environ.get('SLURM_ARRAY_TASK_ID', '0'))
    if not 0 <= index < len(config.INVERSE_ABLATION_PHASE4_CONDITIONS):
        raise ValueError(f'Invalid condition task ID: {index}')
    return config.INVERSE_ABLATION_PHASE4_CONDITIONS[index]


def make_loader(dataset, shuffle):
    return DataLoader(
        dataset,
        batch_size=config.INVERSE_ABLATION_PHASE4_BATCH_SIZE,
        shuffle=shuffle,
        num_workers=config.INVERSE_ABLATION_PHASE4_NUM_WORKERS,
        pin_memory=True,
        persistent_workers=config.INVERSE_ABLATION_PHASE4_NUM_WORKERS > 0,
    )


def training_paths(family, condition):
    """Retries get new logs, but retain the canonical checkpoint directory."""
    version = f'{family}_{condition}_seed{config.INVERSE_ABLATION_SELECTION_SEED}'
    original_log_dir = os.path.join(config.INVERSE_ABLATION_PHASE4_LOG_ROOT, version)
    checkpoint_dir = os.path.join(original_log_dir, 'checkpoints')
    resume_from = None

    if os.environ.get('INVERSE_ABLATION_PHASE4_RETRY') == '1':
        last_checkpoint = os.path.join(checkpoint_dir, 'last.ckpt')
        if os.path.isfile(last_checkpoint):
            resume_from = last_checkpoint
        retry_id = os.environ.get('SLURM_ARRAY_JOB_ID', 'manual')
        version = f'{version}_retry{retry_id}'

    log_dir = os.path.join(config.INVERSE_ABLATION_PHASE4_LOG_ROOT, version)
    return version, log_dir, checkpoint_dir, resume_from


def main():
    family, condition = selected_condition()
    pl.seed_everything(config.INVERSE_ABLATION_SELECTION_SEED, workers=True)
    cpu_count = int(os.environ.get('SLURM_CPUS_PER_TASK', '12'))
    torch.set_num_threads(max(1, cpu_count - config.INVERSE_ABLATION_PHASE4_NUM_WORKERS))

    # Load the configuration-level splits prepared by Phase 4.
    split_root = os.path.join(config.INVERSE_ABLATION_PHASE4_OUTPUT_ROOT, family, condition)
    train_dataset = ManifestDataset(os.path.join(split_root, 'train.csv'))
    validation_dataset = ManifestDataset(os.path.join(split_root, 'validation.csv'))

    if train_dataset.configuration_ids & validation_dataset.configuration_ids:
        raise ValueError('Configuration leakage between training and validation')
    if len(train_dataset) + len(validation_dataset) != config.INVERSE_ABLATION_PHASE2_TOTAL_SAMPLES:
        raise ValueError('Training plus validation must contain exactly 90k pairs')

    # Keep a small run summary alongside the model logs and checkpoints.
    version, log_dir, checkpoint_dir, resume_from = training_paths(family, condition)
    os.makedirs(log_dir, exist_ok=False)  # Never overwrite an existing training run.
    settings = {
        'family': family,
        'condition': condition,
        'seed': config.INVERSE_ABLATION_SELECTION_SEED,
        'train_pairs': len(train_dataset),
        'validation_pairs': len(validation_dataset),
        'train_configurations': len(train_dataset.configuration_ids),
        'validation_configurations': len(validation_dataset.configuration_ids),
        'batch_size': config.INVERSE_ABLATION_PHASE4_BATCH_SIZE,
        'devices': 1,
        'resume_from': resume_from,
        'checkpoint_directory': checkpoint_dir,
        'target_resolution': 'native 32x32; no target resizing',
    }
    with open(os.path.join(log_dir, 'run_settings.json'), 'w') as handle:
        json.dump(settings, handle, indent=2)
    print(json.dumps(settings, indent=2), flush=True)

    train_loader = make_loader(train_dataset, shuffle=True)
    validation_loader = make_loader(validation_dataset, shuffle=False)

    # Reuse the existing VAE/U-Net and loss settings.
    model = UNet_Resblocks(
        in_channels=4,
        use_vae=True,
        features=list(config.INVERSE_ABLATION_PHASE4_FEATURES),
        learning_rate=config.INVERSE_ABLATION_PHASE4_LEARNING_RATE,
        lambda_=config.INVERSE_ABLATION_PHASE4_DICE_WEIGHT,
        pos_weight=torch.tensor([config.INVERSE_ABLATION_PHASE4_POS_WEIGHT]),
        dropout=config.INVERSE_ABLATION_PHASE4_DROPOUT,
        weight_decay=config.INVERSE_ABLATION_PHASE4_WEIGHT_DECAY,
    )
    loggers = [
        TensorBoardLogger(config.INVERSE_ABLATION_PHASE4_LOG_ROOT, name='', version=version),
        CSVLogger(config.INVERSE_ABLATION_PHASE4_LOG_ROOT, name='', version=version),
    ]
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            save_last=True,
            filename='best-{epoch:03d}-{val_loss:.4f}',
        ),
        EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=config.INVERSE_ABLATION_PHASE4_PATIENCE,
        ),
        LearningRateMonitor(logging_interval='epoch'),
    ]
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        precision=32,
        max_epochs=config.INVERSE_ABLATION_PHASE4_MAX_EPOCHS,
        logger=loggers,
        callbacks=callbacks,
    )
    trainer.fit(model, train_loader, validation_loader, ckpt_path=resume_from)


if __name__ == '__main__':
    main()
