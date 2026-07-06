"""
models/fusion/__init__.py
──────────────────────────
Public API for the fusion sub-package.
"""

from models.fusion.cross_attention_fusion import (
    TemporalPositionEncoding,
    TemporalAlignmentModule,
    CrossModalAttentionBlock,
    CrossModalFusion,
    DisasterClassifier,
    DisasterFusionModel,
)

__all__ = [
    "TemporalPositionEncoding",
    "TemporalAlignmentModule",
    "CrossModalAttentionBlock",
    "CrossModalFusion",
    "DisasterClassifier",
    "DisasterFusionModel",
]
