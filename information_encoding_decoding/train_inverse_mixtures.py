"""Train the three mixed-data conditions; reuse Phase 4 endpoint runs."""
import json
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from models import UNet_Resblocks
from prepare_inverse_mixture_splits import mixture_name
from train_inverse_learning_curves import ManifestDataset, make_loader
from utils import config


def selected_percentage():
    index = int(os.environ.get('SLURM_ARRAY_TASK_ID', '0'))
    percentages = config.INVERSE_ABLATION_PHASE5_TRAIN_PERCENTAGES
    if not 0 <= index < len(percentages):
        raise ValueError(f'Invalid mixture task ID: {index}')
    return percentages[index]


def main():
    percentage = selected_percentage()
    condition = mixture_name(percentage)
    pl.seed_everything(config.INVERSE_ABLATION_SELECTION_SEED, workers=True)
    cpu_count = int(os.environ.get('SLURM_CPUS_PER_TASK', '12'))
    torch.set_num_threads(max(1, cpu_count - config.INVERSE_ABLATION_PHASE4_NUM_WORKERS))

    split_root = os.path.join(config.INVERSE_ABLATION_PHASE5_OUTPUT_ROOT, condition)
    train_dataset = ManifestDataset(os.path.join(split_root, 'train.csv'))
    validation_dataset = ManifestDataset(os.path.join(split_root, 'validation.csv'))
    if train_dataset.configuration_ids & validation_dataset.configuration_ids:
        raise ValueError('Configuration leakage between training and validation')
    if len(train_dataset) + len(validation_dataset) != config.INVERSE_ABLATION_PHASE2_TOTAL_SAMPLES:
        raise ValueError('Training plus validation must contain exactly 90k pairs')

    version = f'{condition}_seed{config.INVERSE_ABLATION_SELECTION_SEED}'
    if os.environ.get('INVERSE_ABLATION_RETRY') == '1':
        version += f"_retry{os.environ.get('SLURM_ARRAY_JOB_ID', 'manual')}"
    log_dir = os.path.join(config.INVERSE_ABLATION_PHASE5_LOG_ROOT, version)
    os.makedirs(log_dir, exist_ok=False)
    settings = {
        'condition': condition,
        'experimental_percentage_of_total': percentage,
        'seed': config.INVERSE_ABLATION_SELECTION_SEED,
        'train_pairs': len(train_dataset),
        'validation_pairs': len(validation_dataset),
        'train_configurations': len(train_dataset.configuration_ids),
        'validation_configurations': len(validation_dataset.configuration_ids),
        'batch_size': config.INVERSE_ABLATION_PHASE4_BATCH_SIZE,
        'devices': 1,
    }
    with open(os.path.join(log_dir, 'run_settings.json'), 'w') as handle:
        json.dump(settings, handle, indent=2)
    print(settings, flush=True)

    train_loader = make_loader(train_dataset, shuffle=True)
    validation_loader = make_loader(validation_dataset, shuffle=False)
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
        TensorBoardLogger(config.INVERSE_ABLATION_PHASE5_LOG_ROOT, name='', version=version),
        CSVLogger(config.INVERSE_ABLATION_PHASE5_LOG_ROOT, name='', version=version),
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
    trainer.fit(model, train_loader, validation_loader)


if __name__ == '__main__':
    main()
