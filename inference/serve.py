"""
inference/serve.py
───────────────────
FastAPI REST server for real-time disaster detection inference.
  POST /predict       — single SAR + tweets prediction
  POST /predict/batch — batch prediction
  GET  /health        — health check
  GET  /alerts/recent — recent alerts from PostGIS
  GET  /stats         — model performance stats
"""

import os
import sys
import json
import base64
import asyncio
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.error("FastAPI not installed. pip install fastapi uvicorn")

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.encoders.sar_encoder import SAREncoder
from models.encoders.text_encoder import TextEncoder, TweetWindowEncoder
from models.fusion.cross_attention_fusion import DisasterFusionModel


# ── Pydantic Models ───────────────────────────────────────────────────────────

class TweetInput(BaseModel):
    text:             str
    lat:              float
    lon:              float
    timestamp:        Optional[str] = None
    credibility_score: Optional[float] = None
    followers_count:  Optional[int] = 100
    retweet_count:    Optional[int] = 0
    verified:         Optional[bool] = False

class PredictRequest(BaseModel):
    lat:              float = Field(..., description="SAR acquisition latitude")
    lon:              float = Field(..., description="SAR acquisition longitude")
    timestamp:        str   = Field(..., description="SAR acquisition time (ISO 8601)")
    tweets:           List[TweetInput] = Field(default=[], description="Co-located tweets")
    sar_base64:       Optional[str] = Field(None, description="Base64-encoded SAR patch (optional)")
    window_minutes:   int = Field(30, description="Tweet time window in minutes")

class PredictResponse(BaseModel):
    is_disaster:      bool
    disaster_type:    str
    disaster_type_probs: Dict[str, float]
    severity:         str
    confidence:       float
    lat:              float
    lon:              float
    timestamp:        str
    processing_ms:    float

class BatchPredictRequest(BaseModel):
    requests: List[PredictRequest]

class AlertRecord(BaseModel):
    id:               int
    timestamp:        str
    lat:              float
    lon:              float
    disaster_type:    str
    severity:         str
    confidence:       float


# ── Model Manager ─────────────────────────────────────────────────────────────

class ModelManager:
    """Singleton model manager. Loads model once, serves many requests."""

    TYPE_NAMES = ["flood", "earthquake", "wildfire", "cyclone", "landslide", "none"]
    SEV_NAMES  = ["low", "medium", "high"]
    MAX_TWEETS = 50
    TWEET_DIM  = 512

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model  = None
        self._load_model(model_path)

    def _load_model(self, model_path: str):
        logger.info(f"Loading model from {model_path}...")
        sar_enc  = SAREncoder(backbone="simple_cnn", output_dim=512)
        txt_enc  = TextEncoder(model_name="simple_bow",  output_dim=512)
        config   = {"d_model": 512, "n_heads": 8, "n_layers": 2,
                    "hidden_dim": 256, "dropout": 0.1}
        model    = DisasterFusionModel(sar_enc, txt_enc, config)

        if os.path.exists(model_path):
            try:
                ckpt = torch.load(model_path, map_location=self.device)
                model.load_state_dict(ckpt["model_state"])
                logger.success(f"Loaded model weights (epoch {ckpt.get('epoch', '?')})")
            except Exception as e:
                logger.warning(f"Could not load weights: {e}. Using untrained model.")
        else:
            logger.warning(f"Model not found at {model_path}. Using untrained model.")

        self.model = model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, request: "PredictRequest") -> dict:
        import time
        t0 = time.perf_counter()

        # ── SAR image ─────────────────────────────────────────────────────
        if request.sar_base64:
            # Decode base64 → numpy → tensor
            sar_bytes = base64.b64decode(request.sar_base64)
            sar_array = np.frombuffer(sar_bytes, dtype=np.float32).reshape(2, 256, 256)
        else:
            # Generate synthetic SAR patch (for demo when no real SAR available)
            sar_array = np.random.normal(-8, 4, (2, 256, 256)).astype(np.float32)
            sar_array = np.clip((sar_array + 25) / 30, 0, 1)

        sar_tensor = torch.from_numpy(sar_array).unsqueeze(0).to(self.device)  # (1, 2, 256, 256)

        # ── Tweet embeddings ──────────────────────────────────────────────
        tweets     = request.tweets[:self.MAX_TWEETS]
        n_tweets   = max(len(tweets), 1)

        tweet_embeds = torch.zeros(1, self.MAX_TWEETS, self.TWEET_DIM).to(self.device)
        time_offsets = torch.zeros(1, self.MAX_TWEETS, dtype=torch.long).to(self.device)
        credibility  = torch.zeros(1, self.MAX_TWEETS).to(self.device)
        tweet_mask   = torch.zeros(1, self.MAX_TWEETS, dtype=torch.bool).to(self.device)

        # Encode tweets with BOW encoder
        texts = [t.text for t in tweets] if tweets else ["no data available"]
        embs  = self.model.text_encoder(texts=texts)       # (n_tweets, D)
        tweet_embeds[0, :len(embs)] = embs

        for i, t in enumerate(tweets):
            # Compute time offset (minutes from SAR acquisition)
            try:
                sar_ts   = datetime.fromisoformat(request.timestamp.replace("Z", "+00:00"))
                twt_ts   = datetime.fromisoformat((t.timestamp or request.timestamp).replace("Z", "+00:00"))
                offset   = abs(int((sar_ts - twt_ts).total_seconds() / 60))
            except Exception:
                offset = 0
            time_offsets[0, i] = min(offset, 59)
            credibility[0, i]  = t.credibility_score or 0.5
            tweet_mask[0, i]   = True

        if not tweets:
            tweet_mask[0, 0]  = True
            credibility[0, 0] = 0.1

        # ── Inference ────────────────────────────────────────────────────
        output = self.model(sar_tensor, tweet_embeds, time_offsets, credibility, tweet_mask)

        bin_probs  = F.softmax(output["binary"],   dim=-1)[0]
        type_probs = F.softmax(output["type"],     dim=-1)[0]
        sev_probs  = F.softmax(output["severity"], dim=-1)[0]

        type_idx   = type_probs.argmax().item()
        sev_idx    = sev_probs.argmax().item()
        confidence = float(bin_probs[1].item())
        is_disaster= confidence > 0.5

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return {
            "is_disaster":         is_disaster,
            "disaster_type":       self.TYPE_NAMES[type_idx],
            "disaster_type_probs": {
                name: round(float(type_probs[i].item()), 4)
                for i, name in enumerate(self.TYPE_NAMES)
            },
            "severity":            self.SEV_NAMES[sev_idx],
            "confidence":          round(confidence, 4),
            "lat":                 request.lat,
            "lon":                 request.lon,
            "timestamp":           request.timestamp,
            "processing_ms":       round(elapsed_ms, 2),
        }


# ── FastAPI Application ───────────────────────────────────────────────────────

MODEL_PATH = os.getenv("MODEL_PATH", "checkpoints/best_model.pt")
DEVICE     = os.getenv("INFERENCE_DEVICE", "cpu")

if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="Disaster Detection API",
        description="Real-time SAR + Social Media fusion disaster detection",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Lazy-load model on first request
    _manager: Optional[ModelManager] = None

    def get_manager() -> ModelManager:
        global _manager
        if _manager is None:
            _manager = ModelManager(MODEL_PATH, DEVICE)
        return _manager

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model_loaded": _manager is not None,
            "device": DEVICE,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        manager = get_manager()
        result  = manager.predict(req)
        return PredictResponse(**result)

    @app.post("/predict/batch")
    def predict_batch(req: BatchPredictRequest):
        manager = get_manager()
        results = [manager.predict(r) for r in req.requests]
        return {"predictions": results, "count": len(results)}

    @app.get("/alerts/recent")
    def recent_alerts(hours: int = 24, limit: int = 100):
        """Return recent alerts from PostGIS."""
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            return {"alerts": [], "message": "No database configured"}

        try:
            from sqlalchemy import create_engine, text
            engine = create_engine(db_url)
            since  = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
            with engine.connect() as conn:
                rows = conn.execute(text("""
                    SELECT id, timestamp::text, lat, lon,
                           disaster_type, severity, confidence
                    FROM disaster_alerts
                    WHERE timestamp > :since
                    ORDER BY timestamp DESC
                    LIMIT :limit
                """), {"since": since, "limit": limit})
                alerts = [dict(row._mapping) for row in rows]
            return {"alerts": alerts, "count": len(alerts)}
        except Exception as e:
            return {"alerts": [], "error": str(e)}

    @app.get("/stats")
    def stats():
        """Return model and system stats."""
        return {
            "model_path":    MODEL_PATH,
            "device":        DEVICE,
            "disaster_types": ModelManager.TYPE_NAMES,
            "severity_levels":ModelManager.SEV_NAMES,
            "max_tweets":    ModelManager.MAX_TWEETS,
            "tweet_dim":     ModelManager.TWEET_DIM,
        }


def start_server(host: str = "0.0.0.0", port: int = 8000):
    if not FASTAPI_AVAILABLE:
        logger.error("FastAPI not available. pip install fastapi uvicorn")
        return
    logger.info(f"Starting inference server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    start_server(args.host, args.port)
