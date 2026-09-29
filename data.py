"""Landslide4Sense data loading and preprocessing.

Each Landslide4Sense patch is a 128x128x14 array (Sentinel-2 B1-B12, slope, DEM).
We build a 6-channel input: Red, Green, Blue, NDVI, Slope, Elevation.
"""
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, random_split

# 0-based band indices in the 14-band Landslide4Sense patches
BLUE, GREEN, RED, NIR, SLOPE, DEM = 1, 2, 3, 7, 12, 13
CHANNELS = ('red', 'green', 'blue', 'ndvi', 'slope', 'elevation')

# Per-channel statistics used for the released checkpoints. They were computed
# with compute_mean_std() over all of datasets/images/train (train + val split).
MEAN = np.array([0.95963868, 0.95410915, 0.92270051, 0.01454027, 1.25110812, 1.64954336])
STD = np.array([0.57237459, 0.31842474, 0.22069762, 0.30451106, 0.67836859, 1.07271148])


def load_image(path):
    with h5py.File(path, 'r') as hdf:
        return np.nan_to_num(np.array(hdf.get('img')), nan=0.0)


def load_mask(path):
    with h5py.File(path, 'r') as hdf:
        return np.array(hdf.get('mask')).astype(np.float32)


def build_features(data):
    """(H, W, 14) raw patch -> (6, H, W) un-normalised feature stack."""
    red, green, blue, nir = data[:, :, RED], data[:, :, GREEN], data[:, :, BLUE], data[:, :, NIR]

    denom = nir + red
    denom[denom == 0] = 1e-6
    ndvi = (nir - red) / denom

    return np.stack([red, green, blue, ndvi, data[:, :, SLOPE], data[:, :, DEM]], axis=0)


def normalize(img, mean=MEAN, std=STD):
    return (img - mean[:, None, None]) / std[:, None, None]


def compute_mean_std(image_paths):
    feats = np.stack([build_features(load_image(p)) for p in image_paths])
    return feats.mean(axis=(0, 2, 3)), feats.std(axis=(0, 2, 3))


def mask_name_for(image_name):
    """image_123.h5 -> mask_123.h5"""
    return image_name.replace('image', 'mask', 1)


class LandslideDataset(Dataset):
    """Yields ``(image, mask)``; with ``mask_dir=None`` yields ``(image, file_name)``."""

    def __init__(self, image_dir, mask_dir=None, mean=MEAN, std=STD):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mean = np.asarray(mean)
        self.std = np.asarray(std)
        self.image_files = sorted(f for f in os.listdir(image_dir) if f.endswith('.h5'))

        if mask_dir is not None:
            missing = [f for f in self.image_files
                       if not os.path.exists(os.path.join(mask_dir, mask_name_for(f)))]
            if missing:
                raise FileNotFoundError(
                    f'{len(missing)} images in {image_dir} have no matching mask in {mask_dir}, '
                    f'e.g. {missing[0]} -> {mask_name_for(missing[0])}')

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        name = self.image_files[idx]
        img = normalize(build_features(load_image(os.path.join(self.image_dir, name))), self.mean, self.std)
        img = torch.from_numpy(img).float()

        if self.mask_dir is None:
            return img, name

        mask = load_mask(os.path.join(self.mask_dir, mask_name_for(name)))
        return img, torch.from_numpy(mask).float().unsqueeze(0)


def split_dirs(data_root, split):
    return os.path.join(data_root, 'images', split), os.path.join(data_root, 'annotations', split)


def train_val_split(dataset, val_fraction=0.2, seed=42):
    """The 80/20 split used for the released checkpoints (seed 42).

    Keep ``seed`` fixed across training runs so that repeated runs differ only
    in initialisation and batch order, not in which patches are validation.
    """
    val_size = int(val_fraction * len(dataset))
    return random_split(dataset, [len(dataset) - val_size, val_size],
                        generator=torch.Generator().manual_seed(seed))
