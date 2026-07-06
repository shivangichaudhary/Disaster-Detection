"""
models/classifier/__init__.py
──────────────────────────────
Public API of the models.classifier package.
"""

from models.classifier.disaster_classifier import (
    DisasterClassifierHead,
    DisasterPredictor,
    DisasterClassifier,        # re-exported from fusion module
    DisasterFusionModel,       # re-exported from fusion module
    build_full_model,
    load_checkpoint,
    TYPE_NAMES,
    SEVERITY_NAMES,
    DEFAULT_CONFIG,
    MAX_TWEETS,
    TWEET_DIM,
)

__all__ = [
    "DisasterClassifierHead",
    "DisasterPredictor",
    "DisasterClassifier",
    "DisasterFusionModel",
    "build_full_model",
    "load_checkpoint",
    "TYPE_NAMES",
    "SEVERITY_NAMES",
    "DEFAULT_CONFIG",
    "MAX_TWEETS",
    "TWEET_DIM",
]
