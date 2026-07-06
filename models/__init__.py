"""
models/__init__.py
───────────────────
Top-level package for all model components.
"""

from models.encoders.sar_encoder import SAREncoder, SAREncoderWithAuxHead
from models.encoders.text_encoder import TextEncoder, TweetWindowEncoder
from models.fusion.cross_attention_fusion import (
    TemporalPositionEncoding,
    TemporalAlignmentModule,
    CrossModalAttentionBlock,
    CrossModalFusion,
    DisasterClassifier,
    DisasterFusionModel,
)
from models.classifier.disaster_classifier import (
    DisasterClassifierHead,
    DisasterPredictor,
    build_full_model,
    load_checkpoint,
    TYPE_NAMES,
    SEVERITY_NAMES,
    DEFAULT_CONFIG,
)

__all__ = [
    # Encoders
    "SAREncoder",
    "SAREncoderWithAuxHead",
    "TextEncoder",
    "TweetWindowEncoder",
    # Fusion
    "TemporalPositionEncoding",
    "TemporalAlignmentModule",
    "CrossModalAttentionBlock",
    "CrossModalFusion",
    "DisasterClassifier",
    "DisasterFusionModel",
    # Classifier
    "DisasterClassifierHead",
    "DisasterPredictor",
    "build_full_model",
    "load_checkpoint",
    "TYPE_NAMES",
    "SEVERITY_NAMES",
    "DEFAULT_CONFIG",
]
