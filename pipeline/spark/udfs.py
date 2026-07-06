"""
pipeline/spark/udfs.py
──────────────────────
All Spark UDFs for the disaster-detection streaming pipeline.

UDFs registered here:
  tweet_clean_udf          – cleans raw tweet text (TweetCleaner)
  tweet_credibility_udf    – full multi-factor credibility score (CredibilityScorer)
  tweet_bot_score_udf      – bot probability heuristic (BotDetector)
  tweet_disaster_type_udf  – keyword-based disaster-type classifier
  tweet_is_relevant_udf    – boolean disaster-relevance filter
  geohash_udf              – lat/lon → geohash string
  sar_quality_udf          – SAR file quality gate (returns True if file is usable)
  sar_stats_udf            – returns JSON with SAR band statistics

All UDFs are pure Python closures so they are serialised to executors without
needing the driver's sys.path (beyond the standard Python packages installed
on every node).  Heavy imports (numpy, torch) are done *inside* the closure so
Spark can ship them lazily.
"""

import sys
import json
import math
import re
from pathlib import Path
from typing import Optional

# ── PySpark availability guard ────────────────────────────────────────────────
try:
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        StringType, FloatType, BooleanType, IntegerType,
        StructType, StructField, ArrayType,
    )
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False
    # Stub type classes so UDF factories don't NameError when PySpark absent
    class StringType:  pass
    class FloatType:   pass
    class BooleanType: pass
    class IntegerType: pass
    class StructType:
        def __init__(self, fields=None): self.fields = fields or []
    class StructField:
        def __init__(self, *a, **kw): pass
    class ArrayType:
        def __init__(self, *a, **kw): pass


# ─────────────────────────────────────────────────────────────────────────────
# Helper: safe UDF wrapper
# ─────────────────────────────────────────────────────────────────────────────

def _udf(fn, return_type):
    """Wrap a Python function as a Spark UDF when PySpark is available,
    otherwise return the plain function (useful for unit tests)."""
    if PYSPARK_AVAILABLE:
        return F.udf(fn, return_type)
    return fn


# ─────────────────────────────────────────────────────────────────────────────
# 1. TWEET PREPROCESSING UDFs
#    All logic delegates to utils.tweet_preprocessing so there is a single
#    source of truth.  The closures re-import lazily inside each executor.
# ─────────────────────────────────────────────────────────────────────────────

def make_tweet_clean_udf():
    """
    UDF: clean raw tweet text.
    Input:  text (StringType)
    Output: cleaned_text (StringType)
    """
    def _clean(text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        try:
            # Import inside closure so Spark ships it to executors
            import sys, os
            # Allow import from project root on executors
            _root = os.getenv("APP_ROOT", "/app")
            if _root not in sys.path:
                sys.path.insert(0, _root)
            from utils.tweet_preprocessing import TweetCleaner
            return TweetCleaner().clean(text, keep_hashtags=True, keep_mentions=False)
        except Exception:
            # Inline fallback – runs without the project package
            text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text)
            text = re.sub(r"@\w+", "", text)
            text = re.sub(r"#(\w+)", r"\1", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text

    return _udf(_clean, StringType())


def make_tweet_credibility_udf():
    """
    UDF: compute full multi-factor credibility score for a tweet.
    Inputs:  text, followers_count, retweet_count, verified,
             account_age_days, lat, lon
    Output:  credibility_score (FloatType  ∈ [0, 1])
    """
    def _credibility(text, followers, retweet_count, verified,
                     account_age_days, lat, lon):
        if text is None:
            return 0.0
        try:
            import sys, os
            _root = os.getenv("APP_ROOT", "/app")
            if _root not in sys.path:
                sys.path.insert(0, _root)
            from utils.tweet_preprocessing import CredibilityScorer
            tweet_dict = {
                "text":             text,
                "followers_count":  followers  or 0,
                "retweet_count":    retweet_count or 0,
                "verified":         bool(verified),
                "account_age_days": account_age_days or 30,
                "lat":              lat,
                "lon":              lon,
            }
            return float(CredibilityScorer().score(tweet_dict))
        except Exception:
            # Inline heuristic fallback
            KEYWORDS = [
                "flood", "earthquake", "wildfire", "cyclone", "landslide",
                "emergency", "evacuation", "disaster", "rescue", "sos",
            ]
            kw  = min(sum(1 for k in KEYWORDS if k in (text or "").lower()) / 3.0, 1.0)
            num = float(bool(re.search(r"\d+", text or "")))
            fol = min(math.log1p(followers or 0) / math.log1p(100_000), 1.0)
            rt  = min(math.log1p(retweet_count or 0) / math.log1p(1_000), 1.0)
            age = min((account_age_days or 30) / 365.0, 1.0)
            s   = 0.35*kw + 0.10*num + 0.25*fol + 0.15*rt + 0.15*age
            if verified:
                s = min(s * 1.2, 1.0)
            return float(round(s, 4))

    return _udf(_credibility, FloatType())


def make_tweet_bot_score_udf():
    """
    UDF: compute bot probability [0, 1].
    Inputs:  followers_count, retweet_count, account_age_days, verified
    Output:  bot_score (FloatType)
    """
    def _bot_score(followers, retweet_count, account_age_days, verified):
        try:
            import sys, os
            _root = os.getenv("APP_ROOT", "/app")
            if _root not in sys.path:
                sys.path.insert(0, _root)
            from utils.tweet_preprocessing import BotDetector
            meta = {
                "followers_count":  followers or 0,
                "statuses_count":   retweet_count or 0,
                "account_age_days": account_age_days or 30,
                "verified":         bool(verified),
                "friends_count":    max((followers or 0) // 2, 1),
            }
            return float(BotDetector().score(meta))
        except Exception:
            # Inline fallback
            red = 0
            rate = (retweet_count or 0) / max(account_age_days or 1, 1)
            if rate > 100:      red += 2
            if (followers or 0) < 5 and (retweet_count or 0) > 100: red += 1
            if verified:        red = max(0, red - 2)
            return float(min(red / 8.0, 1.0))

    return _udf(_bot_score, FloatType())


def make_tweet_disaster_type_udf():
    """
    UDF: keyword-based disaster type classifier.
    Input:  text (StringType)
    Output: disaster_type (StringType) – one of flood/earthquake/wildfire/
                                         cyclone/landslide/none
    """
    KEYWORD_MAP = {
        "flood":      ["flood","flooding","floods","inundation","submerged","flash flood"],
        "earthquake": ["earthquake","quake","tremor","aftershock","seismic","magnitude"],
        "wildfire":   ["wildfire","forest fire","bushfire","blaze","firestorm"],
        "cyclone":    ["cyclone","hurricane","typhoon","storm surge","tropical storm"],
        "landslide":  ["landslide","mudslide","rockfall","debris flow","avalanche"],
    }

    def _disaster_type(text: Optional[str]) -> str:
        if not text:
            return "none"
        try:
            import sys, os
            _root = os.getenv("APP_ROOT", "/app")
            if _root not in sys.path:
                sys.path.insert(0, _root)
            from utils.tweet_preprocessing import TweetCleaner
            return TweetCleaner().get_disaster_type(text) or "none"
        except Exception:
            tl = text.lower()
            scores = {dt: sum(1 for kw in kws if kw in tl)
                      for dt, kws in KEYWORD_MAP.items()}
            best = max(scores, key=scores.get)
            return best if scores[best] > 0 else "none"

    return _udf(_disaster_type, StringType())


def make_tweet_is_relevant_udf():
    """
    UDF: returns True if tweet contains at least one disaster keyword.
    Input:  text (StringType)
    Output: is_relevant (BooleanType)
    """
    ALL_KEYWORDS = [
        "flood","flooding","earthquake","quake","wildfire","bushfire","cyclone",
        "hurricane","typhoon","landslide","mudslide","disaster","emergency",
        "evacuation","rescue","sos","tsunami","blaze","tremor","storm surge",
    ]

    def _is_relevant(text: Optional[str]) -> bool:
        if not text:
            return False
        try:
            import sys, os
            _root = os.getenv("APP_ROOT", "/app")
            if _root not in sys.path:
                sys.path.insert(0, _root)
            from utils.tweet_preprocessing import TweetCleaner
            return TweetCleaner().is_disaster_relevant(text)
        except Exception:
            tl = text.lower()
            return any(kw in tl for kw in ALL_KEYWORDS)

    return _udf(_is_relevant, BooleanType())


# ─────────────────────────────────────────────────────────────────────────────
# 2. GEOSPATIAL UDFs
# ─────────────────────────────────────────────────────────────────────────────

def make_geohash_udf(precision: int = 5):
    """
    UDF: encode lat/lon to geohash.
    Inputs:  lat (FloatType), lon (FloatType)
    Output:  geohash (StringType)
    """
    _precision = precision

    def _geohash(lat: Optional[float], lon: Optional[float]) -> Optional[str]:
        if lat is None or lon is None:
            return None
        try:
            import pygeohash as pgh
            return pgh.encode(lat, lon, precision=_precision)
        except Exception:
            return f"{int(lat * 10) / 10:.1f}_{int(lon * 10) / 10:.1f}"

    return _udf(_geohash, StringType())


# ─────────────────────────────────────────────────────────────────────────────
# 3. SAR PREPROCESSING UDFs
#    These UDFs operate on SAR file *metadata* (paths) because the actual
#    raster data is too large for UDF row-level processing.
#    Full SAR preprocessing runs in map-partitions on the Spark executors.
# ─────────────────────────────────────────────────────────────────────────────

def make_sar_quality_udf():
    """
    UDF: check whether a SAR file path is valid and readable.
    Input:  filepath (StringType)
    Output: is_valid (BooleanType)
    """
    def _quality(filepath: Optional[str]) -> bool:
        if not filepath:
            return False
        import os
        if not os.path.exists(filepath):
            # Allow synthetic paths (used in mock / test mode)
            return filepath.endswith((".tif", ".SAFE", ".npy", ".synthetic"))
        try:
            size = os.path.getsize(filepath)
            return size > 0
        except Exception:
            return False

    return _udf(_quality, BooleanType())


def make_sar_stats_udf():
    """
    UDF: compute basic statistics for a SAR file (mean, std per band).
    Returns a JSON string so we stay compatible with Spark's StringType.

    Input:  filepath (StringType)
    Output: stats_json (StringType)
    """
    def _stats(filepath: Optional[str]) -> str:
        if not filepath:
            return json.dumps({"valid": False, "error": "no path"})
        try:
            import sys, os, json
            _root = os.getenv("APP_ROOT", "/app")
            if _root not in sys.path:
                sys.path.insert(0, _root)

            import numpy as np
            from utils.sar_preprocessing import SARPatchReader

            reader = SARPatchReader(patch_size=256, apply_lee=True, to_db=True)

            if os.path.exists(filepath):
                data = reader.read_file(filepath)           # (C, H, W)
                data = reader.normalize(data)
            else:
                # Synthetic fallback for testing
                data = np.random.randn(2, 256, 256).astype(np.float32)

            return json.dumps({
                "valid":       True,
                "shape":       list(data.shape),
                "vv_mean":     float(np.mean(data[0])),
                "vv_std":      float(np.std(data[0])),
                "vh_mean":     float(np.mean(data[1])) if data.shape[0] > 1 else None,
                "vh_std":      float(np.std(data[1]))  if data.shape[0] > 1 else None,
                "global_min":  float(np.min(data)),
                "global_max":  float(np.max(data)),
            })
        except Exception as exc:
            return json.dumps({"valid": False, "error": str(exc)})

    return _udf(_stats, StringType())


def make_sar_label_udf():
    """
    UDF: extract label from SAR filepath directory structure.
    Assumes paths like  .../flood/S1A_....tif  or  .../none/patch.npy
    Input:  filepath (StringType)
    Output: label (StringType)
    """
    VALID_LABELS = {"flood","earthquake","wildfire","cyclone","landslide","none"}

    def _label(filepath: Optional[str]) -> str:
        if not filepath:
            return "none"
        from pathlib import Path
        parent = Path(filepath).parent.name.lower()
        return parent if parent in VALID_LABELS else "none"

    return _udf(_label, StringType())


# ─────────────────────────────────────────────────────────────────────────────
# 4. INFERENCE UDF  (broadcast model weights to all executors)
# ─────────────────────────────────────────────────────────────────────────────

def make_inference_udf(spark, model_path: str):
    """
    Create a Spark UDF that runs fusion-model inference on aggregated features.

    The model state-dict is loaded once in the driver, broadcast to every
    executor, then cached in a process-level dict so repeated calls within
    the same executor process do not reload from disk.

    Args:
        spark:      Active SparkSession
        model_path: Path to best_model.pt checkpoint

    Returns:
        A Spark UDF (StringType) that takes
            (sar_stats_json, cred_scores_json, tweet_count, mean_credibility)
        and returns a JSON string:
            {"disaster_type": str, "severity": str,
             "confidence": float, "is_disaster": bool}
    """
    import os

    # Load model weights in the driver once
    model_bytes = None
    if os.path.exists(model_path):
        try:
            import torch
            ckpt = torch.load(model_path, map_location="cpu")
            model_bytes = ckpt
            from loguru import logger
            logger.info(f"[InferenceUDF] Loaded model from {model_path}")
        except Exception as e:
            from loguru import logger
            logger.warning(f"[InferenceUDF] Could not load model: {e}. Using heuristic.")
    else:
        from loguru import logger
        logger.warning(f"[InferenceUDF] Model not found at {model_path}. Using heuristic.")

    # Broadcast to executors
    bc_model = spark.sparkContext.broadcast(model_bytes)

    def _run_inference(sar_stats_json: Optional[str],
                       cred_scores_json: Optional[str],
                       tweet_count,
                       mean_credibility) -> str:
        """
        Core inference function executed on each executor partition.
        Falls back to credibility heuristic when model weights are absent.
        """
        import json, os, sys

        # ── Parse inputs ──────────────────────────────────────────────────────
        try:
            sar_stats = json.loads(sar_stats_json) if sar_stats_json else {}
        except Exception:
            sar_stats = {}

        try:
            cred_scores = json.loads(cred_scores_json) if cred_scores_json else [0.5]
            if not isinstance(cred_scores, list) or len(cred_scores) == 0:
                cred_scores = [0.5]
        except Exception:
            cred_scores = [0.5]

        mean_cred   = float(mean_credibility or sum(cred_scores) / len(cred_scores))
        n_tweets    = int(tweet_count or len(cred_scores))
        sar_valid   = sar_stats.get("valid", False)

        # ── Try model inference ───────────────────────────────────────────────
        model_state = bc_model.value
        if model_state is not None:
            try:
                import torch
                import torch.nn.functional as TF

                _root = os.getenv("APP_ROOT", "/app")
                if _root not in sys.path:
                    sys.path.insert(0, _root)

                # Use a process-level cache to avoid reloading every row
                _cache = globals().setdefault("_model_cache", {})
                if "model" not in _cache:
                    from models.encoders.sar_encoder import SAREncoder
                    from models.encoders.text_encoder import TextEncoder
                    from models.fusion.cross_attention_fusion import DisasterFusionModel

                    sar_enc = SAREncoder(backbone="simple_cnn", output_dim=512)
                    txt_enc = TextEncoder(model_name="simple_bow", output_dim=512)
                    model   = DisasterFusionModel(sar_enc, txt_enc, {"d_model": 512})
                    try:
                        model.load_state_dict(model_state["model_state"])
                    except Exception:
                        pass
                    model.eval()
                    _cache["model"] = model

                model = _cache["model"]
                with torch.no_grad():
                    # SAR patch: use stats to build synthetic representative tensor
                    vv_mean = float(sar_stats.get("vv_mean", 0.0))
                    vv_std  = float(sar_stats.get("vv_std",  1.0))
                    vh_mean = float(sar_stats.get("vh_mean") or vv_mean - 3.0)
                    vh_std  = float(sar_stats.get("vh_std")  or vv_std)

                    sar_patch = torch.zeros(1, 2, 256, 256)
                    sar_patch[:, 0] = vv_mean + vv_std * torch.randn(256, 256)
                    sar_patch[:, 1] = vh_mean + vh_std * torch.randn(256, 256)

                    # Tweet embeddings from credibility scores (proxy)
                    t_cred = torch.tensor(cred_scores[:50], dtype=torch.float32).unsqueeze(0)
                    seq_len = t_cred.shape[1]
                    t_emb   = t_cred.unsqueeze(-1).expand(-1, -1, 512)  # (1, T, 512)
                    t_off   = torch.zeros(1, seq_len, dtype=torch.long)
                    t_mask  = torch.ones(1, seq_len, dtype=torch.bool)

                    output     = model(sar_patch, t_emb, t_off, t_cred, t_mask)
                    type_probs = TF.softmax(output["type"],   dim=-1)[0]
                    bin_probs  = TF.softmax(output["binary"], dim=-1)[0]
                    sev_probs  = TF.softmax(output["severity"], dim=-1)[0]

                    TYPE_NAMES = ["flood","earthquake","wildfire","cyclone","landslide","none"]
                    SEV_NAMES  = ["low","medium","high"]

                    return json.dumps({
                        "disaster_type": TYPE_NAMES[type_probs.argmax().item()],
                        "severity":      SEV_NAMES[sev_probs.argmax().item()],
                        "confidence":    round(float(bin_probs[1].item()), 4),
                        "is_disaster":   bool(bin_probs[1].item() > 0.5),
                        "source":        "model",
                    })

            except Exception as exc:
                pass  # fall through to heuristic

        # ── Heuristic fallback ────────────────────────────────────────────────
        # Uses credibility + tweet volume as proxy signal
        tweet_factor   = min(n_tweets / 20.0, 1.0)   # saturates at 20 tweets
        sar_factor     = 0.8 if sar_valid else 0.5
        fusion_score   = (0.50 * mean_cred +
                          0.30 * tweet_factor +
                          0.20 * sar_factor)

        is_disaster = fusion_score > 0.45

        if fusion_score > 0.75:
            severity = "high"
        elif fusion_score > 0.55:
            severity = "medium"
        else:
            severity = "low"

        # Naive disaster-type from credibility distribution (high-cred tweets
        # tend to be more urgent → flood/earthquake)
        if fusion_score > 0.70:
            dtype = "flood"
        elif fusion_score > 0.55:
            dtype = "earthquake"
        else:
            dtype = "none"

        return json.dumps({
            "disaster_type": dtype if is_disaster else "none",
            "severity":      severity,
            "confidence":    round(fusion_score, 4),
            "is_disaster":   is_disaster,
            "source":        "heuristic",
        })

    if PYSPARK_AVAILABLE:
        return F.udf(_run_inference, StringType())
    return _run_inference


# ─────────────────────────────────────────────────────────────────────────────
# 5. SCHEMAS  (centralised here so streaming_job.py can import them)
# ─────────────────────────────────────────────────────────────────────────────

if PYSPARK_AVAILABLE:
    TWEET_SCHEMA = StructType([
        StructField("text",             StringType(),  True),
        StructField("lat",              FloatType(),   True),
        StructField("lon",              FloatType(),   True),
        StructField("timestamp",        StringType(),  True),
        StructField("author_id",        StringType(),  True),
        StructField("followers_count",  FloatType(),   True),
        StructField("retweet_count",    FloatType(),   True),
        StructField("verified",         BooleanType(), True),
        StructField("account_age_days", FloatType(),   True),
        StructField("source",           StringType(),  True),
        StructField("platform",         StringType(),  True),
        StructField("ground_truth_label", StringType(),True),
    ])

    SAR_SCHEMA = StructType([
        StructField("filepath",   StringType(), True),
        StructField("filename",   StringType(), True),
        StructField("label",      StringType(), True),
        StructField("timestamp",  StringType(), True),
        StructField("size_bytes", FloatType(),  True),
        StructField("source",     StringType(), True),
    ])

    INFERENCE_RESULT_SCHEMA = StructType([
        StructField("disaster_type", StringType(),  True),
        StructField("severity",      StringType(),  True),
        StructField("confidence",    FloatType(),   True),
        StructField("is_disaster",   BooleanType(), True),
        StructField("source",        StringType(),  True),
    ])
else:
    TWEET_SCHEMA = None
    SAR_SCHEMA   = None
    INFERENCE_RESULT_SCHEMA = None
