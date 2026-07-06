"""
scripts/run_spark_streaming.py
──────────────────────────────
End-to-end launcher for the Disaster Detection streaming pipeline.

What this script does
─────────────────────
  1. Validates environment (PySpark, Kafka, etc.)
  2. Starts a mock Kafka producer in a background thread
     (or connects to a real Bluesky/Mastodon producer if available)
  3. Optionally starts a mock SAR producer for the 'sar-raw' topic
  4. Launches the Spark Structured Streaming job
  5. Prints a live status summary every 30 seconds

Usage
─────
  # Minimal – tweet-only, mock data, local Spark, console output
  python scripts/run_spark_streaming.py

  # With real Bluesky data + Kafka alerts sink
  python scripts/run_spark_streaming.py --source bluesky --mode tweet_only

  # Full fusion mode (tweets + SAR), write results to Postgres
  python scripts/run_spark_streaming.py --mode fusion --source mock --postgres

  # Submit to a real cluster
  SPARK_LOCAL_MODE=false \\
  KAFKA_BOOTSTRAP_SERVERS=broker:9092 \\
  spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1 \\
    scripts/run_spark_streaming.py --mode fusion --source mock

Environment variables
─────────────────────
  KAFKA_BOOTSTRAP_SERVERS   Kafka broker address    (default: localhost:9092)
  DATABASE_URL              PostgreSQL URL           (default: localhost:5432)
  MODEL_PATH                Path to best_model.pt   (default: checkpoints/best_model.pt)
  SPARK_LOCAL_MODE          "true" | "false"         (default: true)
  SPARK_CHECKPOINT_DIR      Checkpoint base dir      (default: /tmp/spark_checkpoints)
  MIN_TWEET_CREDIBILITY     Credibility gate [0,1]   (default: 0.30)
  MAX_BOT_SCORE             Bot score gate   [0,1]   (default: 0.70)
"""

from __future__ import annotations

import os
import sys
import time
import signal
import threading
import argparse
from pathlib import Path
from datetime import datetime

# ── Project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from loguru import logger
from dotenv import load_dotenv
load_dotenv(_ROOT / ".env.example")          # load defaults; real .env overrides


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def check_environment() -> dict:
    """Check availability of all required components."""
    status = {}

    # PySpark
    try:
        import pyspark
        status["pyspark"] = f"✓  PySpark {pyspark.__version__}"
    except ImportError:
        status["pyspark"] = "✗  PySpark NOT installed  →  pip install pyspark>=3.4.0"

    # Kafka
    try:
        from kafka import KafkaProducer
        status["kafka"] = "✓  kafka-python available"
    except ImportError:
        status["kafka"] = "✗  kafka-python NOT installed  →  pip install kafka-python"

    # Kafka broker reachability
    try:
        from kafka import KafkaAdminClient
        from pipeline.spark.spark_config import KAFKA_BOOTSTRAP
        admin = KafkaAdminClient(bootstrap_servers=KAFKA_BOOTSTRAP, request_timeout_ms=3000)
        admin.close()
        status["kafka_broker"] = f"✓  Kafka broker reachable @ {KAFKA_BOOTSTRAP}"
    except Exception as e:
        from pipeline.spark.spark_config import KAFKA_BOOTSTRAP
        status["kafka_broker"] = (
            f"⚠  Kafka broker @ {KAFKA_BOOTSTRAP} not reachable ({type(e).__name__}). "
            f"Using mock/log-only mode."
        )

    # pygeohash
    try:
        import pygeohash
        status["pygeohash"] = "✓  pygeohash available"
    except ImportError:
        status["pygeohash"] = "⚠  pygeohash not installed (geohash fallback will be used)"

    # langdetect
    try:
        import langdetect
        status["langdetect"] = "✓  langdetect available"
    except ImportError:
        status["langdetect"] = "⚠  langdetect not installed (language filter skipped)"

    # Model checkpoint
    from pipeline.spark.spark_config import MODEL_PATH
    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / 1_048_576
        status["model"] = f"✓  Model checkpoint found: {MODEL_PATH} ({size_mb:.1f} MB)"
    else:
        status["model"] = (
            f"⚠  Model checkpoint NOT found at {MODEL_PATH}. "
            f"Heuristic inference will be used."
        )

    return status


def print_status(status: dict):
    """Print environment status summary."""
    print("\n" + "═" * 65)
    print("  Disaster Detection – Spark Streaming  |  Environment Check")
    print("═" * 65)
    for key, msg in status.items():
        print(f"  {msg}")
    print("═" * 65 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# KAFKA TOPIC SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def ensure_kafka_topics():
    """Create required Kafka topics if they don't already exist."""
    try:
        from kafka import KafkaAdminClient
        from kafka.admin import NewTopic
        from pipeline.spark.spark_config import KAFKA_BOOTSTRAP

        admin = KafkaAdminClient(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            request_timeout_ms=5000,
        )
        existing = set(admin.list_topics())
        topics_needed = [
            NewTopic(name="tweets-raw", num_partitions=3, replication_factor=1),
            NewTopic(name="sar-raw",    num_partitions=1, replication_factor=1),
            NewTopic(name="alerts",     num_partitions=3, replication_factor=1),
        ]
        to_create = [t for t in topics_needed if t.name not in existing]
        if to_create:
            admin.create_topics(to_create, validate_only=False)
            logger.info(f"Created Kafka topics: {[t.name for t in to_create]}")
        else:
            logger.info("All required Kafka topics already exist.")
        admin.close()
    except Exception as exc:
        logger.warning(f"Could not auto-create Kafka topics: {exc}. Topics may be auto-created by broker.")


# ═══════════════════════════════════════════════════════════════════════════════
# PRODUCERS  (run in background threads)
# ═══════════════════════════════════════════════════════════════════════════════

def start_tweet_producer(source: str, interval: float) -> threading.Thread | None:
    """Start the social media producer in a background thread."""
    try:
        from pipeline.kafka.producers import KafkaProducerWrapper
        from pipeline.kafka.social_stream_producer import create_producer

        kafka_prod = KafkaProducerWrapper()
        producer   = create_producer(source, kafka_prod)

        if source == "mock":
            producer.interval = interval

        def _run():
            logger.info(f"[Producer] Starting {source} tweet producer → tweets-raw")
            try:
                producer.start()
            except Exception as exc:
                logger.error(f"[Producer] {source} producer crashed: {exc}")

        t = threading.Thread(target=_run, daemon=True, name=f"producer-{source}")
        t.start()
        logger.info(f"[Producer] {source} tweet producer started (thread={t.name})")
        return t
    except Exception as exc:
        logger.warning(f"[Producer] Could not start {source} producer: {exc}")
        return None


def start_sar_producer(
    sar_dir: str = "data/raw/sentinel1",
    poll_interval: int = 30,
    use_mock: bool = True,
    mock_interval: float = 15.0,
) -> threading.Thread | None:
    """
    Start a SAR producer in a background thread.

    Args:
        sar_dir:       Directory to watch for real .tif files (use_mock=False).
        poll_interval: Scan interval for real files (seconds).
        use_mock:      If True, use SARMockProducer (generates synthetic .npy patches).
                       Set to False only if you have real Sentinel-1 .tif files in sar_dir.
        mock_interval: Seconds between synthetic SAR events (use_mock=True).
    """
    try:
        from pipeline.kafka.producers import KafkaProducerWrapper

        kafka_prod = KafkaProducerWrapper()

        if use_mock:
            # ── Mock producer: writes synthetic .npy patches to a temp dir ────
            from pipeline.kafka.sar_mock_producer import SARMockProducer
            sar_prod = SARMockProducer(
                kafka_prod,
                interval=mock_interval,
                disaster_ratio=0.70,
            )

            def _run_mock():
                logger.info(
                    f"[SAR Producer] Mock mode: generating synthetic SAR patches → sar-raw "
                    f"(interval={mock_interval}s)"
                )
                try:
                    sar_prod.start()
                except Exception as exc:
                    logger.error(f"[SAR Producer] Mock producer crashed: {exc}")

            t = threading.Thread(target=_run_mock, daemon=True, name="sar-mock-producer")
            t.start()
            logger.info("[SAR Producer] Mock SAR producer started ✓")
            return t

        else:
            # ── Real file watcher: polls sar_dir for new .tif / .SAFE files ──
            from pipeline.kafka.producers import SARImageProducer
            sar_prod = SARImageProducer(kafka_prod, watch_dir=sar_dir)

            def _run_real():
                logger.info(
                    f"[SAR Producer] Real mode: watching {sar_dir} → sar-raw "
                    f"(every {poll_interval}s)"
                )
                try:
                    sar_prod.start(poll_interval=poll_interval)
                except Exception as exc:
                    logger.error(f"[SAR Producer] Real producer crashed: {exc}")

            t = threading.Thread(target=_run_real, daemon=True, name="sar-real-producer")
            t.start()
            logger.info(f"[SAR Producer] Real file watcher started → {sar_dir} ✓")
            return t

    except Exception as exc:
        logger.warning(f"[SAR Producer] Could not start: {exc}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE STATUS MONITOR
# ═══════════════════════════════════════════════════════════════════════════════

def monitor_queries(spark_ref: list, stop_event: threading.Event):
    """Print streaming query status every 30 seconds."""
    while not stop_event.is_set():
        time.sleep(30)
        if stop_event.is_set():
            break
        try:
            spark = spark_ref[0]
            active = spark.streams.active
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Active queries: {len(active)}")
            for q in active:
                progress = q.lastProgress
                if progress:
                    inp = progress.get("numInputRows", 0)
                    rate = progress.get("inputRowsPerSecond", 0.0)
                    print(f"  {q.name:30s}  rows={inp}  rate={rate:.1f}/s")
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Disaster Detection – Spark Streaming Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        choices=["mock", "bluesky", "mastodon", "twitter"],
        default="mock",
        help="Social media data source. 'mock' always works offline. (default: mock)",
    )
    parser.add_argument(
        "--mode",
        choices=["tweet_only", "fusion"],
        default="tweet_only",
        help="'tweet_only': tweets-only pipeline. 'fusion': SAR + tweets. (default: tweet_only)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.5,
        help="Mock producer interval in seconds (default: 1.5)",
    )
    parser.add_argument(
        "--sar-dir",
        default="data/raw/sentinel1",
        help="Directory to watch for real Sentinel-1 .tif files (--real-sar mode only)",
    )
    parser.add_argument(
        "--sar-interval",
        type=float,
        default=15.0,
        help="Mock SAR event interval in seconds (default: 15.0)",
    )
    parser.add_argument(
        "--real-sar",
        action="store_true",
        help="Use real SAR files from --sar-dir instead of synthetic mock patches",
    )
    parser.add_argument(
        "--no-kafka",
        action="store_true",
        help="Disable Kafka alert sink (alerts will not be written to 'alerts' topic)",
    )
    parser.add_argument(
        "--postgres",
        action="store_true",
        help="Enable PostgreSQL sink (requires PostGIS running)",
    )
    parser.add_argument(
        "--no-console",
        action="store_true",
        help="Disable console sink (suppress micro-batch output)",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip environment validation",
    )
    args = parser.parse_args()

    # ── Banner ────────────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  🌊  Disaster Detection – Apache Spark Streaming Pipeline")
    print(f"  Mode: {args.mode}  |  Source: {args.source}")
    print("═" * 65)

    # ── Environment check ─────────────────────────────────────────────────────
    if not args.skip_check:
        status = check_environment()
        print_status(status)
        if "✗" in " ".join(status.values()):
            logger.warning("Some required packages are missing. Pipeline may not function correctly.")

    # ── Kafka topic creation ───────────────────────────────────────────────────
    ensure_kafka_topics()

    # ── Background producers ──────────────────────────────────────────────────
    threads = []

    tweet_thread = start_tweet_producer(args.source, args.interval)
    if tweet_thread:
        threads.append(tweet_thread)

    if args.mode == "fusion":
        sar_thread = start_sar_producer(
            sar_dir=args.sar_dir,
            poll_interval=30,
            use_mock=not args.real_sar,
            mock_interval=args.sar_interval,
        )
        if sar_thread:
            threads.append(sar_thread)

    # Give producers a moment to connect / generate first messages
    logger.info("Waiting 3 s for producers to warm up...")
    time.sleep(3)

    # ── Spark streaming job ───────────────────────────────────────────────────
    try:
        from pipeline.spark.streaming_job import run_streaming_pipeline
    except ImportError as exc:
        logger.error(f"Could not import streaming job: {exc}")
        logger.error("Make sure PySpark is installed: pip install pyspark>=3.4.0")
        sys.exit(1)

    logger.info(f"[Launcher] Starting Spark streaming pipeline  (mode={args.mode})")

    # Run the pipeline (this blocks until Ctrl-C or query failure)
    run_streaming_pipeline(
        mode=args.mode,
        enable_kafka_sink=not args.no_kafka,
        enable_postgres_sink=args.postgres,
        enable_console_sink=not args.no_console,
        await_termination=True,
    )


if __name__ == "__main__":
    main()
