"""
models/classifier/disaster_classifier.py
─────────────────────────────────────────
Standalone classifier interface for the disaster detection system.

Provides:
  • DisasterClassifierHead  – lightweight multi-task classification heads
    (binary / type / severity).  Mirrors the DisasterClassifier defined
    inside cross_attention_fusion.py so either can be imported.

  • build_full_model()       – convenience factory: constructs the complete
    DisasterFusionModel (SAR encoder + text encoder + TAM + fusion +
    classifier) ready for training or inference.

  • DisasterPredictor        – high-level wrapper around DisasterFusionModel
    that accepts raw numpy / list inputs and returns structured prediction
    dicts.  Used by inference/serve.py and the Spark streaming job.

Disaster types  : flood | earthquake | wildfire | cyclone | landslide | none
Severity levels : low | medium | high
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

# ── make sure the project root is on sys.path ─────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.encoders.sar_encoder import SAREncoder
from models.encoders.text_encoder import TextEncoder
from models.fusion.cross_attention_fusion import (
    DisasterFusionModel,
    DisasterClassifier,   # re-export for convenience
)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

TYPE_NAMES: List[str]     = ["flood", "earthquake", "wildfire", "cyclone", "landslide", "none"]
SEVERITY_NAMES: List[str] = ["low", "medium", "high"]

# Default model hyper-parameters (must match training config)
DEFAULT_CONFIG: Dict = {
    "d_model":    512,
    "n_heads":    8,
    "n_heads_tam": 4,
    "n_layers":   2,
    "hidden_dim": 256,
    "dropout":    0.1,
    "backbone":   "simple_cnn",   # SAREncoder backbone
    "text_model": "simple_bow",   # TextEncoder variant
}

MAX_TWEETS: int = 50   # maximum tweet-window size accepted by the model
TWEET_DIM:  int = 512  # must match TextEncoder output_dim


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DisasterClassifierHead  (alias for the head in cross_attention_fusion)
# ─────────────────────────────────────────────────────────────────────────────

class DisasterClassifierHead(nn.Module):
    """
    Standalone multi-task classification head.

    Input:  fused representation  (B, d_model)
    Output: dict with logits
        'binary'   (B, 2)   – disaster vs no-disaster
        'type'     (B, 6)   – flood / earthquake / wildfire / cyclone / landslide / none
        'severity' (B, 3)   – low / medium / high

    This class is intentionally identical to DisasterClassifier in
    cross_attention_fusion.py so callers can use either import path.
    """

    TYPE_LABELS:     List[str] = TYPE_NAMES
    SEVERITY_LABELS: List[str] = SEVERITY_NAMES

    def __init__(
        self,
        d_model:    int   = 512,
        hidden_dim: int   = 256,
        dropout:    float = 0.3,
    ):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_binary   = nn.Linear(hidden_dim, 2)
        self.head_type     = nn.Linear(hidden_dim, len(self.TYPE_LABELS))
        self.head_severity = nn.Linear(hidden_dim, len(self.SEVERITY_LABELS))

    def forward(self, fused: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            fused: (B, d_model) fused SAR+text representation
        Returns:
            dict with keys 'binary', 'type', 'severity' — all logits
        """
        h = self.shared(fused)
        return {
            "binary":   self.head_binary(h),
            "type":     self.head_type(h),
            "severity": self.head_severity(h),
        }

    # ── Convenience ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_probs(self, fused: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return softmax probabilities instead of raw logits."""
        logits = self.forward(fused)
        return {k: F.softmax(v, dim=-1) for k, v in logits.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Factory: build the full end-to-end model
# ─────────────────────────────────────────────────────────────────────────────

def build_full_model(config: Optional[Dict] = None) -> DisasterFusionModel:
    """
    Construct the full DisasterFusionModel from scratch.

    Args:
        config: optional override dict.  Missing keys fall back to
                DEFAULT_CONFIG values.

    Returns:
        Untrained DisasterFusionModel instance.

    Example::

        model = build_full_model()
        model = build_full_model({"n_layers": 4, "dropout": 0.2})
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    sar_encoder  = SAREncoder(
        backbone=cfg.get("backbone", "simple_cnn"),
        output_dim=cfg["d_model"],
    )
    text_encoder = TextEncoder(
        model_name=cfg.get("text_model", "simple_bow"),
        output_dim=cfg["d_model"],
    )

    model = DisasterFusionModel(sar_encoder, text_encoder, cfg)
    logger.debug(
        f"Built DisasterFusionModel | d_model={cfg['d_model']} "
        f"n_layers={cfg['n_layers']} backbone={cfg['backbone']}"
    )
    return model


def load_checkpoint(
    model_path: str,
    config:     Optional[Dict] = None,
    device:     str            = "cpu",
) -> DisasterFusionModel:
    """
    Build a model and load weights from a checkpoint file.

    Args:
        model_path: path to .pt checkpoint saved by training/train.py
        config:     optional model config overrides
        device:     'cpu' or 'cuda'

    Returns:
        Model with loaded weights set to eval() mode.
    """
    dev   = torch.device(device)
    model = build_full_model(config)

    if os.path.exists(model_path):
        try:
            ckpt = torch.load(model_path, map_location=dev)
            model.load_state_dict(ckpt["model_state"])
            epoch = ckpt.get("epoch", "?")
            logger.success(f"Loaded checkpoint from '{model_path}' (epoch {epoch})")
        except Exception as exc:
            logger.warning(f"Could not load weights from '{model_path}': {exc}. Using random init.")
    else:
        logger.warning(f"No checkpoint at '{model_path}'. Using randomly initialised model.")

    return model.to(dev).eval()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  DisasterPredictor — high-level inference wrapper
# ─────────────────────────────────────────────────────────────────────────────

class DisasterPredictor:
    """
    High-level wrapper around DisasterFusionModel for production inference.

    Accepts raw Python/numpy inputs and returns structured dicts.
    Thread-safe (model is in eval mode, no state mutation during predict).

    Usage::

        predictor = DisasterPredictor.from_checkpoint("checkpoints/best_model.pt")

        result = predictor.predict(
            sar_patch=np.random.randn(2, 256, 256).astype("float32"),
            tweets=[{"text": "flooding!", "lat": 19.0, "lon": 72.8,
                     "credibility_score": 0.8, "time_offset_min": 5}],
        )
        print(result["disaster_type"], result["confidence"])
    """

    def __init__(self, model: DisasterFusionModel, device: str = "cpu"):
        self.device = torch.device(device)
        self.model  = model.to(self.device).eval()

    # ── Constructors ─────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        model_path: str,
        config:     Optional[Dict] = None,
        device:     str            = "cpu",
    ) -> "DisasterPredictor":
        model = load_checkpoint(model_path, config, device)
        return cls(model, device)

    @classmethod
    def from_scratch(
        cls,
        config: Optional[Dict] = None,
        device: str            = "cpu",
    ) -> "DisasterPredictor":
        model = build_full_model(config).to(device).eval()
        return cls(model, device)

    # ── Core inference ────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        sar_patch:  Optional[np.ndarray] = None,
        tweets:     Optional[List[Dict]] = None,
        lat:        float                = 0.0,
        lon:        float                = 0.0,
        timestamp:  str                  = "",
    ) -> Dict:
        """
        Run inference for a single (SAR patch, tweet window) pair.

        Args:
            sar_patch:  numpy array of shape (2, 256, 256) — dual-pol SAR.
                        If None a synthetic patch is generated.
            tweets:     list of tweet dicts, each with keys:
                          text             (str)
                          credibility_score (float, 0-1)  default 0.5
                          time_offset_min  (int)          default 0
            lat, lon:   event location (informational, passed through)
            timestamp:  ISO-8601 string (informational, passed through)

        Returns:
            dict with keys:
                is_disaster        bool
                disaster_type      str
                disaster_type_probs dict[str, float]
                severity           str
                confidence         float   (P(disaster))
                lat, lon           float
                timestamp          str
        """
        import time as _time
        t0 = _time.perf_counter()

        # ── SAR tensor ───────────────────────────────────────────────────────
        if sar_patch is None:
            sar_array = np.random.normal(-8, 4, (2, 256, 256)).astype(np.float32)
            sar_array = np.clip((sar_array + 25) / 30, 0.0, 1.0)
        else:
            sar_array = np.asarray(sar_patch, dtype=np.float32)
            if sar_array.ndim == 2:
                sar_array = np.stack([sar_array, sar_array])  # mono → dual-pol
        sar_t = torch.from_numpy(sar_array).unsqueeze(0).to(self.device)  # (1, 2, 256, 256)

        # ── Tweet tensors ─────────────────────────────────────────────────────
        tweets = tweets or []
        tweets = tweets[:MAX_TWEETS]

        tweet_embeds = torch.zeros(1, MAX_TWEETS, TWEET_DIM, device=self.device)
        time_offsets = torch.zeros(1, MAX_TWEETS, dtype=torch.long,  device=self.device)
        credibility  = torch.zeros(1, MAX_TWEETS,                    device=self.device)
        tweet_mask   = torch.zeros(1, MAX_TWEETS, dtype=torch.bool,  device=self.device)

        if tweets:
            texts = [t.get("text", "") for t in tweets]
            embs  = self.model.text_encoder(texts=texts)  # (N, D)
            tweet_embeds[0, :len(embs)] = embs

            for i, tw in enumerate(tweets):
                time_offsets[0, i] = min(int(tw.get("time_offset_min", 0)), 59)
                credibility[0, i]  = float(tw.get("credibility_score", 0.5))
                tweet_mask[0, i]   = True
        else:
            # No tweets — seed a weak single slot so TAM has something to attend to
            tweet_mask[0, 0]  = True
            credibility[0, 0] = 0.1

        # ── Forward pass ──────────────────────────────────────────────────────
        output = self.model(sar_t, tweet_embeds, time_offsets, credibility, tweet_mask)

        bin_probs  = F.softmax(output["binary"],   dim=-1)[0]
        type_probs = F.softmax(output["type"],     dim=-1)[0]
        sev_probs  = F.softmax(output["severity"], dim=-1)[0]

        type_idx    = int(type_probs.argmax())
        sev_idx     = int(sev_probs.argmax())
        confidence  = float(bin_probs[1])
        is_disaster = confidence > 0.5

        elapsed_ms  = (_time.perf_counter() - t0) * 1000

        return {
            "is_disaster":         is_disaster,
            "disaster_type":       TYPE_NAMES[type_idx],
            "disaster_type_probs": {
                name: round(float(type_probs[i]), 4)
                for i, name in enumerate(TYPE_NAMES)
            },
            "severity":            SEVERITY_NAMES[sev_idx],
            "confidence":          round(confidence, 4),
            "lat":                 lat,
            "lon":                 lon,
            "timestamp":           timestamp,
            "processing_ms":       round(elapsed_ms, 2),
        }

    def predict_batch(
        self,
        samples: List[Dict],
    ) -> List[Dict]:
        """
        Run predict() for each sample in *samples*.

        Each element of *samples* must be a dict accepted by predict()
        (keys: sar_patch, tweets, lat, lon, timestamp — all optional).

        Returns a list of prediction dicts in the same order.
        """
        return [self.predict(**s) for s in samples]


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Module-level convenience re-exports
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    # Classes
    "DisasterClassifierHead",
    "DisasterPredictor",
    # Re-exports from fusion module
    "DisasterClassifier",
    "DisasterFusionModel",
    # Factory functions
    "build_full_model",
    "load_checkpoint",
    # Constants
    "TYPE_NAMES",
    "SEVERITY_NAMES",
    "DEFAULT_CONFIG",
    "MAX_TWEETS",
    "TWEET_DIM",
]


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Running disaster_classifier smoke-test …")

    # 1. Build model
    model = build_full_model()
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # 2. Test standalone head
    head  = DisasterClassifierHead(d_model=512, hidden_dim=256)
    fused = torch.randn(4, 512)
    out   = head(fused)
    assert out["binary"].shape   == (4, 2),  "binary head shape mismatch"
    assert out["type"].shape     == (4, 6),  "type head shape mismatch"
    assert out["severity"].shape == (4, 3),  "severity head shape mismatch"
    logger.success("DisasterClassifierHead: OK")

    # 3. Test DisasterPredictor
    predictor = DisasterPredictor.from_scratch()

    # — no SAR, no tweets
    r1 = predictor.predict(lat=19.0, lon=72.8, timestamp="2024-01-01T00:00:00Z")
    assert r1["disaster_type"] in TYPE_NAMES
    assert 0.0 <= r1["confidence"] <= 1.0
    logger.success(f"predict() (no data): {r1['disaster_type']} conf={r1['confidence']:.3f}")

    # — with SAR patch + tweets
    sar  = np.clip(np.random.normal(-8, 4, (2, 256, 256)).astype("float32") / 20, 0, 1)
    twts = [
        {"text": "Flooding everywhere!", "credibility_score": 0.9, "time_offset_min": 2},
        {"text": "Roads are underwater", "credibility_score": 0.7, "time_offset_min": 10},
    ]
    r2   = predictor.predict(sar_patch=sar, tweets=twts, lat=19.0, lon=72.8)
    assert r2["severity"] in SEVERITY_NAMES
    logger.success(f"predict() (SAR+tweets): type={r2['disaster_type']} sev={r2['severity']}")

    # — batch
    batch_results = predictor.predict_batch([
        {"lat": 28.6, "lon": 77.2},
        {"sar_patch": sar, "tweets": twts, "lat": 13.0, "lon": 80.3},
    ])
    assert len(batch_results) == 2
    logger.success(f"predict_batch(): {len(batch_results)} results")

    logger.success("All smoke-tests passed ✓")
