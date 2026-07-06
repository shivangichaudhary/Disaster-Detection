"""
utils/sar_preprocessing.py
──────────────────────────
SAR (Sentinel-1) image preprocessing pipeline:
  • Speckle filtering (Lee filter)
  • Radiometric calibration  (σ⁰ in dB)
  • Patch extraction (256×256)
  • Normalisation
"""

import numpy as np
import os
from pathlib import Path
from typing import Tuple, List, Optional
import torch
from torch import Tensor
from torchvision import transforms
from tqdm import tqdm
from loguru import logger

try:
    import rasterio
    from rasterio.windows import Window
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    logger.warning("rasterio not available. SAR reading will be limited.")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# ── Lee Speckle Filter ────────────────────────────────────────────────────────

def lee_filter(image: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """
    Apply Lee speckle filter to SAR image.
    Based on: Lee, J.S. (1980) Digital image enhancement and noise filtering.

    Args:
        image: 2D numpy array (single band SAR)
        kernel_size: filter window size (odd integer)
    Returns:
        Filtered image of same shape
    """
    if CV2_AVAILABLE:
        img_sq = image ** 2
        mean   = cv2.blur(image,    (kernel_size, kernel_size))
        mean_sq= cv2.blur(img_sq,   (kernel_size, kernel_size))
    else:
        from scipy.ndimage import uniform_filter
        img_sq  = image ** 2
        mean    = uniform_filter(image,  size=kernel_size)
        mean_sq = uniform_filter(img_sq, size=kernel_size)

    variance   = mean_sq - mean ** 2
    img_var    = np.var(image)
    noise_var  = np.mean(variance)

    # Lee filter weight
    weight = variance / (variance + noise_var + 1e-8)
    filtered = mean + weight * (image - mean)
    return filtered.astype(np.float32)


def apply_lee_filter_multiband(image: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """Apply Lee filter to each band independently. image: (C, H, W)"""
    return np.stack([lee_filter(image[c], kernel_size) for c in range(image.shape[0])])


# ── Radiometric Calibration ───────────────────────────────────────────────────

def linear_to_db(image: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Convert linear backscatter to dB: σ⁰_dB = 10 * log10(σ⁰)"""
    return 10.0 * np.log10(np.maximum(image, eps))


def db_to_linear(image_db: np.ndarray) -> np.ndarray:
    """Convert dB backscatter to linear."""
    return 10.0 ** (image_db / 10.0)


def calibrate_sar(image: np.ndarray, to_db: bool = True) -> np.ndarray:
    """
    Radiometric normalisation for Sentinel-1 GRD data.
    Assumes input is raw DN (digital number) values.
    """
    # σ⁰ = DN² / calibration_constant (approx for GRD)
    sigma0 = image.astype(np.float32) ** 2
    if to_db:
        sigma0 = linear_to_db(sigma0)
    return sigma0


# ── SAR Dataset Reader ────────────────────────────────────────────────────────

class SARPatchReader:
    """
    Reads Sentinel-1 GRD .tif files and returns preprocessed patches.
    Supports both real Sentinel-1 data and our synthetic patches.
    """

    def __init__(
        self,
        patch_size: int = 256,
        bands: List[str] = ["VV", "VH"],
        apply_lee: bool = True,
        to_db: bool = True,
        overlap: float = 0.1,
    ):
        self.patch_size  = patch_size
        self.bands       = bands
        self.apply_lee   = apply_lee
        self.to_db       = to_db
        self.overlap     = overlap
        self.stride      = int(patch_size * (1 - overlap))

    def read_file(self, filepath: str) -> Optional[np.ndarray]:
        """Read a SAR GeoTIFF and return (C, H, W) float32 array."""
        if not RASTERIO_AVAILABLE:
            raise RuntimeError("rasterio required for SAR file reading.")

        with rasterio.open(filepath) as src:
            # Read all bands (VV=band1, VH=band2 for Sentinel-1 GRD)
            data = src.read().astype(np.float32)  # (C, H, W)

        # Speckle filter
        if self.apply_lee:
            data = apply_lee_filter_multiband(data)

        # Convert to dB
        if self.to_db:
            data = linear_to_db(data)

        return data

    def extract_patches(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Slide a window over (C, H, W) image and extract (C, patch_size, patch_size) patches.
        """
        _, H, W = image.shape
        patches = []
        for y in range(0, H - self.patch_size + 1, self.stride):
            for x in range(0, W - self.patch_size + 1, self.stride):
                patch = image[:, y:y+self.patch_size, x:x+self.patch_size]
                patches.append(patch)
        return patches

    def normalize(self, patch: np.ndarray) -> np.ndarray:
        """
        Per-band min-max normalisation to [0, 1].
        SAR dB values typically in range [-25, 5] dB.
        """
        db_min, db_max = -25.0, 5.0
        return np.clip((patch - db_min) / (db_max - db_min), 0, 1)


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class SARDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for SAR patches.
    Supports both file-based loading (real data) and in-memory (synthetic).

    Args:
        metadata_df: DataFrame with columns [filepath, label_id, lat, lon, timestamp]
        patch_size: output patch size in pixels
        augment: apply random augmentations during training
    """

    LABEL_NAMES = ["flood", "earthquake", "wildfire", "cyclone", "landslide", "none"]

    def __init__(self, metadata_df, patch_size: int = 256, augment: bool = False):
        self.df         = metadata_df.reset_index(drop=True)
        self.reader     = SARPatchReader(patch_size=patch_size)
        self.patch_size = patch_size
        self.augment    = augment
        self._transform = self._build_transform(augment)

    def _build_transform(self, augment: bool) -> transforms.Compose:
        ops = [transforms.ToTensor()]
        if augment:
            ops += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=90),
            ]
        ops.append(
            transforms.Normalize(mean=[0.485, 0.456], std=[0.229, 0.224])
        )
        return transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # Load patch
        if RASTERIO_AVAILABLE and os.path.exists(str(row["filepath"])):
            data = self.reader.read_file(str(row["filepath"]))
        else:
            # Fallback: generate synthetic SAR patch
            data = self._synthetic_patch(str(row.get("label", "none")))

        data = self.reader.normalize(data)

        # Convert (C, H, W) float32 to tensor
        tensor = torch.from_numpy(data)

        # Ensure shape (2, patch_size, patch_size)
        if tensor.shape[0] > 2:
            tensor = tensor[:2]
        elif tensor.shape[0] < 2:
            tensor = tensor.repeat(2, 1, 1)[:2]

        return {
            "image":     tensor,                              # (2, 256, 256)
            "label":     int(row.get("label_id", 5)),
            "lat":       float(row.get("lat", 0.0)),
            "lon":       float(row.get("lon", 0.0)),
            "timestamp": str(row.get("timestamp", "")),
        }

    def _synthetic_patch(self, label: str) -> np.ndarray:
        """Generate a synthetic SAR patch matching the class statistics."""
        stats = {
            "flood":      (-15.0, 3.0),
            "earthquake": (-8.0,  5.0),
            "wildfire":   (-10.0, 4.0),
            "cyclone":    (-12.0, 6.0),
            "landslide":  (-9.0,  4.5),
            "none":       (-6.0,  3.0),
        }
        m, s = stats.get(label, (-8.0, 4.0))
        vv = np.random.normal(m, s, (self.patch_size, self.patch_size)).astype(np.float32)
        vh = (vv + np.random.normal(-3.0, 1.5, vv.shape)).astype(np.float32)
        return np.stack([vv, vh])


# ── Preprocessing Pipeline ────────────────────────────────────────────────────

def preprocess_sar_directory(
    input_dir: str,
    output_dir: str,
    patch_size: int = 256,
    max_files: Optional[int] = None,
):
    """
    Batch preprocess all .tif files in input_dir.
    Saves preprocessed patches as .npy files.
    """
    import pandas as pd

    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    reader   = SARPatchReader(patch_size=patch_size)
    tif_files= list(input_path.rglob("*.tif"))
    if max_files:
        tif_files = tif_files[:max_files]

    metadata = []
    logger.info(f"Processing {len(tif_files)} SAR files...")

    for fpath in tqdm(tif_files, desc="SAR preprocessing"):
        try:
            label  = fpath.parent.name
            data   = reader.read_file(str(fpath))
            data   = reader.normalize(data)
            patches= reader.extract_patches(data) if data.shape[1] > patch_size else [data]

            for i, patch in enumerate(patches):
                out_name = f"{fpath.stem}_patch{i:04d}.npy"
                np.save(output_path / out_name, patch)
                metadata.append({
                    "filepath": str(output_path / out_name),
                    "label": label,
                    "label_id": SARDataset.LABEL_NAMES.index(label)
                    if label in SARDataset.LABEL_NAMES else 5,
                    "source_file": str(fpath),
                    "patch_idx": i,
                })
        except Exception as e:
            logger.warning(f"Failed to process {fpath}: {e}")

    meta_df = pd.DataFrame(metadata)
    meta_df.to_csv(output_path / "sar_patches_metadata.csv", index=False)
    logger.success(f"Saved {len(metadata)} patches to {output_path}")
    return meta_df


if __name__ == "__main__":
    # Quick test with synthetic data
    import pandas as pd

    logger.info("Testing SAR preprocessing pipeline...")

    # Generate synthetic patches
    records = []
    for label_id, label in enumerate(SARDataset.LABEL_NAMES):
        for i in range(5):
            records.append({
                "filepath": f"dummy_{label}_{i}.tif",
                "label": label,
                "label_id": label_id,
                "lat": 20.0 + i,
                "lon": 70.0 + i,
                "timestamp": "2023-09-01T00:00:00",
            })

    df = pd.DataFrame(records)
    ds = SARDataset(df, patch_size=256, augment=True)
    sample = ds[0]

    logger.success(
        f"SAR patch shape: {sample['image'].shape}, "
        f"label: {sample['label']}, "
        f"dtype: {sample['image'].dtype}"
    )
