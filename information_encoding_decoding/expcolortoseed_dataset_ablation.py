from torch.utils.data import Dataset
import os

import cv2
import numpy as np
import torch
from natsort import natsorted

from utils.config import (
    SPECIFIC_FOLDER_EXP_DIFFUSION_MORE,
    SPECIFIC_FOLDER_SEED_DIFFUSION_MORE,
)
from utils.preprocess import (
    preprocess_ablation_bilateral,
    preprocess_ablation_binary,
    preprocess_ablation_gaussian,
    preprocess_ablation_grayscale,
    preprocess_experimental_backgroundwhite,
    preprocess_seed,
)


ABLATION_TYPES = ("none", "grayscale", "binary", "gaussian", "bilateral")


def apply_ablation(image, ablation_type="none", gaussian_sigma=None):
    """Apply one ablation to a final-size RGB uint8 image."""
    if ablation_type not in ABLATION_TYPES:
        raise ValueError(
            f"Unknown ablation_type '{ablation_type}'. Choose from {ABLATION_TYPES}."
        )

    if ablation_type == "none":
        return image
    if ablation_type == "grayscale":
        return preprocess_ablation_grayscale(image)
    if ablation_type == "binary":
        return preprocess_ablation_binary(image)
    if ablation_type == "gaussian":
        return preprocess_ablation_gaussian(image, gaussian_sigma)
    return preprocess_ablation_bilateral(image)


class MyDataset(Dataset):
    def __init__(
        self,
        source_folder=SPECIFIC_FOLDER_EXP_DIFFUSION_MORE,
        target_folder=SPECIFIC_FOLDER_SEED_DIFFUSION_MORE,
        start=0,
        end=None,
        use_vae=False,
        preprocess_method=cv2.INTER_NEAREST,
        preprocess_input="backgroundwhite",
        ablation_type="none",
        gaussian_sigma=None,
        target_preprocessed=False,
    ):
        if end is None:
            end = len(os.listdir(source_folder))

        if ablation_type == "gaussian" and gaussian_sigma is None:
            raise ValueError("gaussian_sigma is required for the gaussian ablation")

        self.use_vae = use_vae
        self.source_files = natsorted(
            [os.path.join(source_folder, filename) for filename in os.listdir(source_folder)]
        )[start:end]
        self.target_files = natsorted(
            [os.path.join(target_folder, filename) for filename in os.listdir(target_folder)]
        )[start:end]
        self.preprocess_method = preprocess_method
        self.preprocess_input = preprocess_input
        self.ablation_type = ablation_type
        self.gaussian_sigma = gaussian_sigma
        self.target_preprocessed = target_preprocessed

        assert len(self.source_files) == len(self.target_files), (
            f"Mismatch: {len(self.source_files)} sources vs "
            f"{len(self.target_files)} targets"
        )

    def __len__(self):
        return len(self.source_files)

    def __getitem__(self, idx):
        source_path = self.source_files[idx]
        target_path = self.target_files[idx]

        if self.preprocess_input == "backgroundwhite":
            source = preprocess_experimental_backgroundwhite(source_path)
        else:
            source = cv2.imread(source_path, cv2.IMREAD_COLOR)
            if source is None:
                raise FileNotFoundError(f"Failed to load source image: {source_path}")
            source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
            source = cv2.resize(source, (256, 256))

        source = apply_ablation(
            source,
            ablation_type=self.ablation_type,
            gaussian_sigma=self.gaussian_sigma,
        )

        if self.target_preprocessed:
            # Native targets are already cropped correctly; no threshold/resize.
            target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
            if target is None or target.shape != (32, 32):
                raise ValueError(f'Expected a native 32x32 seed target: {target_path}')
        elif self.use_vae is False:
            target = preprocess_seed(target_path, method=self.preprocess_method)
        else:
            target = preprocess_seed(target_path, img_length=32, img_width=32)

        source = source.astype("float32") / 255.0
        target = target.astype("float32") / 255.0

        source = np.transpose(source, (2, 0, 1))
        target = target[np.newaxis, :, :]

        source = torch.from_numpy(source).float()
        target = torch.from_numpy(target).float()

        filename = os.path.basename(source_path)
        stem = os.path.splitext(filename)[0]

        return {"source": source, "target": target, "stem": stem}
