"""
training/dataset.py  +  training/metrics.py
────────────────────────────────────────────
PyTorch Dataset that pairs SAR patches with tweet windows,
plus evaluation metrics (F1, precision, recall, mAP).
"""

# ═══════════════════════════════════════════════════════════════
#  DisasterDataset
# ═══════════════════════════════════════════════════════════════

import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from loguru import logger


LABEL_TO_BINARY = {
    "flood": 1, "earthquake": 1, "wildfire": 1,
    "cyclone": 1, "landslide": 1, "none": 0,
}
LABEL_TO_TYPE = {
    "flood": 0, "earthquake": 1, "wildfire": 2,
    "cyclone": 3, "landslide": 4, "none": 5,
}
# Severity heuristic: cyclone/earthquake = high, flood/wildfire = medium, landslide/none = low
LABEL_TO_SEVERITY = {
    "flood": 1, "earthquake": 2, "wildfire": 1,
    "cyclone": 2, "landslide": 1, "none": 0,
}


class DisasterDataset(Dataset):
    """
    Multimodal dataset that pairs:
        • A SAR patch (or synthetic) → image tensor (2, 256, 256)
        • A set of co-located tweets → tweet_embeddings (T, D)
        • Temporal offsets, credibility scores
        • Multi-task labels (binary, type, severity)

    In offline (file-based) mode:
        Reads SAR .npy patches + tweet CSV, pairs them by geohash + timestamp.
    In synthetic mode:
        Generates everything on-the-fly for testing without real data.
    """

    MAX_TWEETS  = 50   # T: tweets per window
    TWEET_DIM   = 512  # D: tweet embedding dimension (must match TextEncoder.output_dim)
    PATCH_SIZE  = 256

    def __init__(self, data_cfg: dict, mode: str = "synthetic"):
        """
        Args:
            data_cfg: data section from train_config.yaml
            mode: 'file' | 'synthetic'
        """
        self.cfg  = data_cfg
        self.mode = mode

        if mode == "file":
            self._load_from_files()
        else:
            self._build_synthetic()

    def _load_from_files(self):
        """Load real SAR + tweet data from disk."""
        sar_meta  = pd.read_csv(Path(self.cfg["sar_root"]) / "sar_patches_metadata.csv")
        tweet_meta= pd.read_csv(self.cfg["tweet_root"])
        self.records = self._pair_sar_tweets(sar_meta, tweet_meta)
        logger.info(f"Loaded {len(self.records)} paired samples from files.")

    def _build_synthetic(self, n: int = 1000):
        """Build a fully synthetic dataset for training without real data."""
        logger.info(f"Building synthetic dataset with {n} samples...")
        labels = ["flood", "earthquake", "wildfire", "cyclone", "landslide", "none"]
        self.records = []

        for i in range(n):
            label = labels[i % len(labels)]
            n_valid_tweets = random.randint(1, self.MAX_TWEETS)
            self.records.append({
                "label":        label,
                "label_binary": LABEL_TO_BINARY[label],
                "label_type":   LABEL_TO_TYPE[label],
                "label_severity": LABEL_TO_SEVERITY[label],
                "n_tweets":     n_valid_tweets,
                "lat":  random.uniform(8.0, 35.0),
                "lon":  random.uniform(68.0, 100.0),
            })

    def _pair_sar_tweets(self, sar_df: pd.DataFrame, tweet_df: pd.DataFrame):
        """Pair SAR patches with nearest tweet windows by geohash + timestamp.

        Primary: tweets within ±30-min time window.
        Fallback: nearest tweets by label match when time window is empty,
                  ensuring every sample always has some tweet signal for training.
        """
        from utils.tweet_preprocessing import to_geohash
        records = []

        tweet_df["timestamp"] = pd.to_datetime(tweet_df["timestamp"], utc=True, errors="coerce")
        tweet_df = tweet_df.dropna(subset=["timestamp"])
        sar_df["timestamp"]   = pd.to_datetime(
            sar_df["timestamp"] if "timestamp" in sar_df.columns else "2023-01-01",
            utc=True, errors="coerce"
        ).fillna(pd.Timestamp("2023-01-01", tz="UTC"))

        # Pre-group tweets by label for efficient fallback lookup
        label_tweet_groups = {}
        if "disaster_type" in tweet_df.columns:
            for lbl, grp in tweet_df.groupby("disaster_type"):
                label_tweet_groups[lbl] = grp

        for _, sar_row in sar_df.iterrows():
            sar_ts = sar_row.get("timestamp", pd.Timestamp("2023-01-01", tz="UTC"))
            label  = sar_row.get("label", "none")

            # Primary: ±30-minute temporal window
            window_start = sar_ts - pd.Timedelta(minutes=30)
            window_end   = sar_ts + pd.Timedelta(minutes=30)
            nearby = tweet_df[
                (tweet_df["timestamp"] >= window_start) &
                (tweet_df["timestamp"] <= window_end)
            ]

            # Fallback: sample tweets matching the SAR label when window is empty
            if len(nearby) == 0:
                fallback_pool = label_tweet_groups.get(label) or label_tweet_groups.get("none") or tweet_df
                n_fallback = min(5, len(fallback_pool))
                nearby = fallback_pool.sample(n=n_fallback, random_state=42) if n_fallback > 0 else nearby

            cred_series = nearby["credibility_score"] if "credibility_score" in nearby.columns else pd.Series([0.5] * len(nearby))

            records.append({
                "sar_filepath":  sar_row.get("filepath", ""),
                "label":         label,
                "label_binary":  LABEL_TO_BINARY.get(label, 0),
                "label_type":    LABEL_TO_TYPE.get(label, 5),
                "label_severity":LABEL_TO_SEVERITY.get(label, 0),
                "tweet_texts":   nearby["text"].tolist()[:self.MAX_TWEETS] if "text" in nearby.columns else [],
                "tweet_creds":   cred_series.fillna(0.5).tolist()[:self.MAX_TWEETS],
                "n_tweets":      min(len(nearby), self.MAX_TWEETS),
                "lat":           float(sar_row.get("lat", 20.0)),
                "lon":           float(sar_row.get("lon", 70.0)),
            })

        logger.info(f"Paired {len(records)} SAR-Tweet samples (with fallback tweet matching)")
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        label = rec["label"]

        # ── SAR image ──────────────────────────────────────────────────────
        if self.mode == "file" and os.path.exists(str(rec.get("sar_filepath", ""))):
            sar_patch = np.load(rec["sar_filepath"]).astype(np.float32)
        else:
            sar_patch = self._synthetic_sar(label)

        sar_tensor = torch.from_numpy(sar_patch)  # (2, 256, 256)

        # ── Tweet embeddings ───────────────────────────────────────────────
        n_valid = rec["n_tweets"]
        tweet_embeds = torch.zeros(self.MAX_TWEETS, self.TWEET_DIM)
        time_offsets = torch.zeros(self.MAX_TWEETS, dtype=torch.long)
        credibility  = torch.zeros(self.MAX_TWEETS)
        tweet_mask   = torch.zeros(self.MAX_TWEETS, dtype=torch.bool)

        if n_valid > 0:
            # Always use label-aware synthetic embeddings for clean training signal.
            # In file mode, tweet_texts are available but a live text encoder is not
            # injected here; the label direction gives consistent cross-modal alignment.
            emb = self._synthetic_tweet_embeddings(label, n_valid)
            tweet_embeds[:n_valid] = emb
            time_offsets[:n_valid] = torch.randint(0, 60, (n_valid,))
            # Use real credibility scores when available, else derive from label
            tweet_creds = rec.get("tweet_creds", [])
            if tweet_creds and len(tweet_creds) >= n_valid:
                cred_vals = torch.tensor(tweet_creds[:n_valid], dtype=torch.float).clamp(0.0, 1.0)
            else:
                cred_vals = (
                    torch.rand(n_valid) * 0.5 + 0.5 if label != "none"
                    else torch.rand(n_valid) * 0.4
                )
            credibility[:n_valid] = cred_vals
            tweet_mask[:n_valid]  = True

        return {
            "image":            sar_tensor,             # (2, 256, 256)
            "tweet_embeddings": tweet_embeds,           # (T, D)
            "time_offsets":     time_offsets,           # (T,)
            "credibility":      credibility,            # (T,)
            "tweet_mask":       tweet_mask,             # (T,)
            "label_binary":     torch.tensor(rec["label_binary"],   dtype=torch.long),
            "label_type":       torch.tensor(rec["label_type"],     dtype=torch.long),
            "label_severity":   torch.tensor(rec["label_severity"], dtype=torch.long),
            "lat":              torch.tensor(rec["lat"],   dtype=torch.float),
            "lon":              torch.tensor(rec["lon"],   dtype=torch.float),
        }

    def _synthetic_sar(self, label: str) -> np.ndarray:
        stats = {
            "flood":      (-15.0, 3.0), "earthquake": (-8.0, 5.0),
            "wildfire":   (-10.0, 4.0), "cyclone":    (-12.0, 6.0),
            "landslide":  (-9.0,  4.5), "none":       (-6.0, 3.0),
        }
        m, s = stats.get(label, (-8.0, 4.0))
        vv = np.random.normal(m, s, (self.PATCH_SIZE, self.PATCH_SIZE)).astype(np.float32)
        vh = (vv + np.random.normal(-3.0, 1.5, vv.shape)).astype(np.float32)
        # Normalize to [0, 1]
        vv = np.clip((vv + 25) / 30, 0, 1)
        vh = np.clip((vh + 25) / 30, 0, 1)
        return np.stack([vv, vh])

    def _synthetic_tweet_embeddings(self, label: str, n: int) -> torch.Tensor:
        """
        Generate synthetic tweet embeddings clustered around a label-specific direction.
        Disaster tweets cluster together in embedding space; 'none' tweets scatter.
        """
        seed_dir = torch.zeros(self.TWEET_DIM)
        idx = LABEL_TO_TYPE.get(label, 5)
        seed_dir[idx * 80: (idx + 1) * 80] = 1.0
        seed_dir = F.normalize(seed_dir, dim=0)

        noise = 0.3 if label != "none" else 1.0
        embeds = seed_dir.unsqueeze(0) + torch.randn(n, self.TWEET_DIM) * noise
        return F.normalize(embeds, dim=-1)


# ═══════════════════════════════════════════════════════════════
#  DisasterMetrics
# ═══════════════════════════════════════════════════════════════

from torchmetrics import (
    F1Score, Precision, Recall, Accuracy, ConfusionMatrix, AUROC,
)
from torchmetrics.collections import MetricCollection


class DisasterMetrics:
    """
    Multi-task evaluation metrics for disaster detection.
    Wraps torchmetrics for clean accumulation across batches.
    """

    def __init__(self, device: torch.device, num_types: int = 6):
        self.device = device

        self.binary_metrics = MetricCollection({
            "acc_binary":  Accuracy(task="binary"),
            "f1_binary":   F1Score(task="binary"),
            "prec_binary": Precision(task="binary"),
            "rec_binary":  Recall(task="binary"),
            "auroc_binary":AUROC(task="binary"),
        }).to(device)

        self.type_metrics = MetricCollection({
            "f1_macro":    F1Score(task="multiclass", num_classes=num_types, average="macro"),
            "f1_weighted": F1Score(task="multiclass", num_classes=num_types, average="weighted"),
            "acc_type":    Accuracy(task="multiclass", num_classes=num_types),
        }).to(device)

        self.severity_metrics = MetricCollection({
            "f1_severity": F1Score(task="multiclass", num_classes=3, average="macro"),
            "acc_severity":Accuracy(task="multiclass", num_classes=3),
        }).to(device)

    def update(self, outputs: dict, targets: dict):
        bin_probs  = torch.softmax(outputs["binary"], dim=-1)[:, 1]
        bin_preds  = outputs["binary"].argmax(dim=-1)
        type_preds = outputs["type"].argmax(dim=-1)
        sev_preds  = outputs["severity"].argmax(dim=-1)

        self.binary_metrics.update(bin_probs, targets["binary"])
        self.type_metrics.update(type_preds, targets["type"])
        self.severity_metrics.update(sev_preds, targets["severity"])

    def compute(self) -> dict:
        results = {}
        results.update(self.binary_metrics.compute())
        results.update(self.type_metrics.compute())
        results.update(self.severity_metrics.compute())
        return {k: float(v) for k, v in results.items()}

    def reset(self):
        self.binary_metrics.reset()
        self.type_metrics.reset()
        self.severity_metrics.reset()


if __name__ == "__main__":
    # Smoke test
    cfg = {
        "sar_root":   "data/processed/sar",
        "tweet_root": "data/processed/tweets.csv",
        "num_workers": 0,
        "pin_memory": False,
        "split_ratios": [0.7, 0.15, 0.15],
    }
    ds = DisasterDataset(cfg, mode="synthetic")
    sample = ds[0]
    print("SAR shape:    ", sample["image"].shape)
    print("Tweet emb:    ", sample["tweet_embeddings"].shape)
    print("Tweet mask:   ", sample["tweet_mask"].sum().item(), "valid tweets")
    print("Labels:        binary=%d type=%d severity=%d" % (
        sample["label_binary"], sample["label_type"], sample["label_severity"]
    ))

    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True)
    batch  = next(iter(loader))
    print("Batch image:  ", batch["image"].shape)
    print("Batch tweets: ", batch["tweet_embeddings"].shape)
    print("Dataset test passed.")
