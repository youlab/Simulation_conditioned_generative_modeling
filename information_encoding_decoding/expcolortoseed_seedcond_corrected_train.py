"""Seed-conditioned inverse training with corrected 3/2 targets and grouped split."""
import csv
import json
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from expcolortoseed_dataset import MyDataset
from models import UNet_Resblocks
from train_inverse_learning_curves import make_loader
from utils import config


class SeedConditionedDataset(MyDataset):
    """Swap predictions, retaining the corrected targets and configuration split."""

    def __init__(self, manifest_path, prediction_files):
        with open(manifest_path, newline='') as handle:
            rows = list(csv.DictReader(handle))

        self.source_files = []
        for row in rows:
            filename = f"Input_{row['configuration_id']}_{row['replicate_id']}.png"
            if filename not in prediction_files:
                raise FileNotFoundError(f'Missing seed-conditioned prediction: {filename}')
            self.source_files.append(os.path.join(
                config.SPECIFIC_FOLDER_EXP_DIFFUSION_MORE_SEEDCOND, filename))

        # Direct 3-left/2-right crops from the original seeds, saved at 32x32.
        self.target_files = [row['target_path'] for row in rows]
        self.configuration_ids = {row['configuration_id'] for row in rows}
        self.use_vae = True
        self.preprocess_input = 'preprocessed'  # Preserve old seed-conditioned input handling.
        self.target_preprocessed = True


def main():
    pl.seed_everything(config.INVERSE_ABLATION_SELECTION_SEED, workers=True)
    cpu_count = int(os.environ.get('SLURM_CPUS_PER_TASK', '12'))
    torch.set_num_threads(max(1, cpu_count - config.INVERSE_ABLATION_PHASE4_NUM_WORKERS))

    # Match the simulation-conditioned N30000 split exactly, not the old sorted split.
    split_root = os.path.join(config.INVERSE_ABLATION_PHASE4_OUTPUT_ROOT, 'synthetic', 'N30000')
    prediction_files = set(config.list_seed_conditioned_predictions())
    train_dataset = SeedConditionedDataset(os.path.join(split_root, 'train.csv'), prediction_files)
    validation_dataset = SeedConditionedDataset(os.path.join(split_root, 'validation.csv'), prediction_files)
    if train_dataset.configuration_ids & validation_dataset.configuration_ids:
        raise ValueError('Configuration leakage between training and validation')
    if len(train_dataset) + len(validation_dataset) != config.INVERSE_ABLATION_PHASE2_TOTAL_SAMPLES:
        raise ValueError('Training plus validation must contain exactly 90k pairs')

    version = f'N30000_seed{config.INVERSE_ABLATION_SELECTION_SEED}'
    log_dir = os.path.join(config.INVERSE_ABLATION_SEEDCOND_LOG_ROOT, version)
    os.makedirs(log_dir, exist_ok=False)
    settings = {
        'prediction_folder': config.SPECIFIC_FOLDER_EXP_DIFFUSION_MORE_SEEDCOND,
        'target_source': 'N30000 native 32x32 targets; direct 3-left/2-right crop',
        'split_source': split_root,
        'train_pairs': len(train_dataset),
        'validation_pairs': len(validation_dataset),
        'train_configurations': len(train_dataset.configuration_ids),
        'validation_configurations': len(validation_dataset.configuration_ids),
        'seed': config.INVERSE_ABLATION_SELECTION_SEED,
        'batch_size': config.INVERSE_ABLATION_PHASE4_BATCH_SIZE,
        'max_epochs': config.INVERSE_ABLATION_PHASE4_MAX_EPOCHS,
        'devices': 1,
        'input_preprocessing': 'RGB resize only; no additional experimental crop',
    }
    with open(os.path.join(log_dir, 'run_settings.json'), 'w') as handle:
        json.dump(settings, handle, indent=2)
    print(settings, flush=True)

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
        TensorBoardLogger(config.INVERSE_ABLATION_SEEDCOND_LOG_ROOT, name='', version=version),
        CSVLogger(config.INVERSE_ABLATION_SEEDCOND_LOG_ROOT, name='', version=version),
    ]
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(log_dir, 'checkpoints'),
            monitor='val_loss', mode='min', save_top_k=1, save_last=True,
            filename='best-{epoch:03d}-{val_loss:.4f}',
        ),
        EarlyStopping(
            monitor='val_loss', mode='min',
            patience=config.INVERSE_ABLATION_PHASE4_PATIENCE,
        ),
        LearningRateMonitor(logging_interval='epoch'),
    ]
    trainer = pl.Trainer(
        accelerator='gpu', devices=1, precision=32,
        max_epochs=config.INVERSE_ABLATION_PHASE4_MAX_EPOCHS,
        logger=loggers, callbacks=callbacks,
    )
    trainer.fit(
        model,
        make_loader(train_dataset, shuffle=True),
        make_loader(validation_dataset, shuffle=False),
    )


if __name__ == '__main__':
    main()
