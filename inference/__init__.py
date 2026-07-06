"""
inference/__init__.py
──────────────────────
Public API for the inference package.

Provides the FastAPI application and model manager for
real-time disaster detection inference.

Usage:
    # Start the server
    python -m uvicorn inference.serve:app --host 0.0.0.0 --port 8000

    # Or programmatically:
    from inference.serve import app, start_server
    start_server(port=8000)
"""

from inference.serve import (
    ModelManager,
    PredictRequest,
    PredictResponse,
    BatchPredictRequest,
    AlertRecord,
    TweetInput,
    start_server,
)

__all__ = [
    "ModelManager",
    "TweetInput",
    "PredictRequest",
    "PredictResponse",
    "BatchPredictRequest",
    "AlertRecord",
    "start_server",
]
