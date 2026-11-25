import json
import os

import cv2
import numpy as np
import torch
from natsort import natsorted
from torch.utils.data import Dataset

from utils.config import SPECIFIC_FOLDER_EXP_DIFFUSION_MORE
from utils.preprocess import preprocess_experimental_backgroundwhite


class MyDataset(Dataset):
    """Diffusion-pattern images paired with colony morphology metrics."""

    def __init__(
        self,
        source_folder=SPECIFIC_FOLDER_EXP_DIFFUSION_MORE,
        metrics_json_path=None,
        start=0,
        end=None,
        image_size=256,
        preprocess_input="None",
    ):
        self.source_folder = source_folder
        self.image_size = image_size
        self.preprocess_input = preprocess_input

        if metrics_json_path is None:
            parent_folder = os.path.dirname(os.path.abspath(source_folder))
            metrics_json_path = os.path.join(parent_folder, "colony_metrics.json")

        with open(metrics_json_path, "r", encoding="utf-8") as file_handle:
            metrics_data = json.load(file_handle)

        records = metrics_data["images"]
        records = natsorted(records, key=lambda record: record["image"]["filename"])
        self.records = records[start:end]

        if not self.records:
            raise ValueError("No image records were selected from the colony metrics JSON file")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        filename = record["image"]["filename"]
        source_path = os.path.join(self.source_folder, filename)

        if self.preprocess_input == "backgroundwhite":
            # Experimental evaluation images: crop the plate border and convert
            # the background to the representation used by the earlier pipeline.
            source = preprocess_experimental_backgroundwhite(
                source_path,
                img_length=self.image_size,
                img_width=self.image_size,
            )
        else:
            # Generated diffusion images used for training are already 256 x 256
            # and are kept uncropped to match their existing metric labels.
            source = cv2.imread(source_path, cv2.IMREAD_COLOR)
            if source is None:
                raise FileNotFoundError(f"Could not read source image: {source_path}")
            source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
            source = cv2.resize(source, (self.image_size, self.image_size))

        if source is None:
            raise FileNotFoundError(f"Could not preprocess source image: {source_path}")

        # The experimental preprocessor crops after its initial resize. Resize
        # the cropped result back to the same 256 x 256 shape used for training.
        source = cv2.resize(source, (self.image_size, self.image_size))
        source = source.astype(np.float32) / 255.0
        source = np.transpose(source, (2, 0, 1))
        source = torch.from_numpy(source)

        metrics = record["metrics"]
        colony_area_fraction = torch.tensor(
            metrics["colony_area_fraction"], dtype=torch.float32
        )
        number_of_tips = torch.tensor(
            metrics["number_of_tips"], dtype=torch.float32
        )

        stem = os.path.splitext(filename)[0]
        return {
            "source": source,
            "colony_area_fraction": colony_area_fraction,
            "number_of_tips": number_of_tips,
            "stem": stem,
        }
