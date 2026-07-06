"""
training/__init__.py
─────────────────────
Public API for the training package.
"""

from training.dataset import (
    DisasterDataset,
    DisasterMetrics,
    LABEL_TO_BINARY,
    LABEL_TO_TYPE,
    LABEL_TO_SEVERITY,
)

from training.train import (
    FocalLoss,
    MultiTaskLoss,
    get_cosine_schedule_with_warmup,
    train_epoch,
    validate,
    train,
)

from training.evaluate import (
    load_model,
    run_evaluation,
    print_results_table,
    save_results,
)

__all__ = [
    # Dataset
    "DisasterDataset",
    "DisasterMetrics",
    "LABEL_TO_BINARY",
    "LABEL_TO_TYPE",
    "LABEL_TO_SEVERITY",
    # Training
    "FocalLoss",
    "MultiTaskLoss",
    "get_cosine_schedule_with_warmup",
    "train_epoch",
    "validate",
    "train",
    # Evaluation
    "load_model",
    "run_evaluation",
    "print_results_table",
    "save_results",
]
