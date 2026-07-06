"""
pipeline/spark/spark_config.py
──────────────────────────────
Centralised SparkSession factory for the disaster-detection streaming pipeline.

Packages bundled at runtime:
  • spark-sql-kafka: Structured Streaming ↔ Kafka
  • postgres JDBC:   foreachBatch writes to PostGIS

Usage:
    from pipeline.spark.spark_config import create_spark_session, KAFKA_BOOTSTRAP, CHECKPOINT_BASE
    spark = create_spark_session()
"""

import os
from pathlib import Path

# ── Environment ───────────────────────────────────────────────────────────────

KAFKA_BOOTSTRAP   = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
DATABASE_URL      = os.getenv(
    "DATABASE_URL",
    "postgresql://disaster:disaster123@localhost:5432/disaster_db"
)
MODEL_PATH        = os.getenv("MODEL_PATH", "checkpoints/best_model.pt")
CHECKPOINT_BASE   = os.getenv("SPARK_CHECKPOINT_DIR", "/tmp/spark_checkpoints")

# Kafka topics
TOPIC_TWEETS_RAW  = "tweets-raw"
TOPIC_SAR_RAW     = "sar-raw"
TOPIC_ALERTS      = "alerts"

# Spark tuning
SHUFFLE_PARTITIONS = int(os.getenv("SPARK_SHUFFLE_PARTITIONS", "8"))
EXECUTOR_MEMORY    = os.getenv("SPARK_EXECUTOR_MEMORY", "4g")
DRIVER_MEMORY      = os.getenv("SPARK_DRIVER_MEMORY",   "2g")

# Kafka connector version (must match Spark version)
_KAFKA_PKG    = "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1"
_POSTGRES_PKG = "org.postgresql:postgresql:42.6.0"


def create_spark_session(app_name: str = "DisasterDetectionStreaming",
                         local_mode: bool = False):
    """
    Build and return a configured SparkSession.

    Args:
        app_name:   Application name shown in Spark UI.
        local_mode: If True, force local[*] master (useful for unit tests /
                    running without a cluster).
    """
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        raise RuntimeError(
            "PySpark not installed. Run: pip install pyspark>=3.4.0"
        )

    builder = (
        SparkSession.builder
        .appName(app_name)
        # ── Kafka + Postgres JDBC drivers ────────────────────────────────────
        .config("spark.jars.packages", f"{_KAFKA_PKG},{_POSTGRES_PKG}")
        # ── Checkpointing ────────────────────────────────────────────────────
        .config("spark.sql.streaming.checkpointLocation", CHECKPOINT_BASE)
        # ── Shuffle ──────────────────────────────────────────────────────────
        .config("spark.sql.shuffle.partitions",           str(SHUFFLE_PARTITIONS))
        # ── Memory ───────────────────────────────────────────────────────────
        .config("spark.executor.memory",  EXECUTOR_MEMORY)
        .config("spark.driver.memory",    DRIVER_MEMORY)
        # ── Serialization (Kryo is faster than Java default) ─────────────────
        .config("spark.serializer",       "org.apache.spark.serializer.KryoSerializer")
        # ── Streaming micro-batch trigger (500 ms) ───────────────────────────
        .config("spark.sql.streaming.statefulOperator.checkCorrectness.enabled", "false")
        # ── UDF optimisation ─────────────────────────────────────────────────
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    )

    if local_mode:
        builder = builder.master("local[*]")

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark
