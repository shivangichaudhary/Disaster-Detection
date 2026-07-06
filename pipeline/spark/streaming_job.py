"""
pipeline/spark/streaming_job.py
────────────────────────────────
Apache Spark Structured Streaming pipeline – Disaster Detection System
======================================================================

Architecture
------------
  [Kafka: tweets-raw] ──► Tweet Stream ──► TweetPreprocessor UDFs
                                                  │
                                                  ▼
                                          Window Aggregation (30-min)
                                                  │
  [Kafka: sar-raw]    ──► SAR Stream   ──► SARPreprocessor UDFs
                                                  │
                                                  ▼
                                          Stream-Stream Join (geohash + 1-hr window)
                                                  │
                                                  ▼
                                          Inference UDF (fusion model / heuristic)
                                                  │
                              ┌───────────────────┼────────────────────┐
                              ▼                   ▼                    ▼
                       [Kafka: alerts]    [PostgreSQL]          [Console debug]

Preprocessing UDFs Used
-----------------------
  Tweet side:
    • tweet_clean_udf          – TweetCleaner.clean()
    • tweet_credibility_udf    – CredibilityScorer.score()
    • tweet_bot_score_udf      – BotDetector.score()
    • tweet_disaster_type_udf  – TweetCleaner.get_disaster_type()
    • tweet_is_relevant_udf    – TweetCleaner.is_disaster_relevant()
    • geohash_udf              – pygeohash.encode()

  SAR side:
    • sar_quality_udf          – file existence + size gate
    • sar_stats_udf            – SARPatchReader (Lee filter → dB → normalise)
    • sar_label_udf            – directory-based label extraction

  Fusion:
    • inference_udf            – broadcast DisasterFusionModel weights

Usage
-----
  # Stand-alone (local mode, no cluster needed)
  python -m pipeline.spark.streaming_job

  # With a real Spark cluster
  spark-submit \\
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1,org.postgresql:postgresql:42.6.0 \\
    pipeline/spark/streaming_job.py
"""

from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Allow running from the disaster_detection package root
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from loguru import logger

# ── PySpark guard ─────────────────────────────────────────────────────────────
try:
    from pyspark.sql import SparkSession, DataFrame
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        StructType, StructField,
        StringType, FloatType, BooleanType,
    )
    from pyspark.sql.streaming import StreamingQuery
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False
    logger.warning("PySpark not installed. Install with: pip install pyspark>=3.4.0")

# ── Local modules ─────────────────────────────────────────────────────────────
from pipeline.spark.spark_config import (
    create_spark_session,
    KAFKA_BOOTSTRAP,
    DATABASE_URL,
    MODEL_PATH,
    CHECKPOINT_BASE,
    TOPIC_TWEETS_RAW,
    TOPIC_SAR_RAW,
    TOPIC_ALERTS,
)
from pipeline.spark.udfs import (
    make_tweet_clean_udf,
    make_tweet_credibility_udf,
    make_tweet_bot_score_udf,
    make_tweet_disaster_type_udf,
    make_tweet_is_relevant_udf,
    make_geohash_udf,
    make_sar_quality_udf,
    make_sar_stats_udf,
    make_sar_label_udf,
    make_inference_udf,
    TWEET_SCHEMA,
    SAR_SCHEMA,
    INFERENCE_RESULT_SCHEMA,
)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – TWEET STREAM
# ═══════════════════════════════════════════════════════════════════════════════

def read_tweet_stream(spark: SparkSession) -> DataFrame:
    """
    Read raw tweets from Kafka topic 'tweets-raw'.
    Deserialises JSON payload using TWEET_SCHEMA.
    """
    logger.info(f"[TweetStream] Connecting to Kafka @ {KAFKA_BOOTSTRAP}, topic={TOPIC_TWEETS_RAW}")
    raw = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe",              TOPIC_TWEETS_RAW)
        .option("startingOffsets",        "latest")
        .option("maxOffsetsPerTrigger",   10_000)
        .option("failOnDataLoss",         "false")
        .load()
    )
    return (
        raw.select(
            F.from_json(F.col("value").cast("string"), TWEET_SCHEMA).alias("d"),
            F.col("timestamp").alias("kafka_ts"),
            F.col("partition").alias("kafka_partition"),
            F.col("offset").alias("kafka_offset"),
        )
        .select("d.*", "kafka_ts", "kafka_partition", "kafka_offset")
    )


def preprocess_tweet_stream(tweet_df: DataFrame) -> DataFrame:
    """
    Apply the full Tweet preprocessing pipeline using UDFs that delegate to
    utils/tweet_preprocessing.py:

      1. Disaster relevance filter  (tweet_is_relevant_udf)
      2. Bot score filter           (tweet_bot_score_udf)
      3. Text cleaning              (tweet_clean_udf)
      4. Credibility scoring        (tweet_credibility_udf)
      5. Credibility threshold gate (> config min_credibility)
      6. Disaster type tagging      (tweet_disaster_type_udf)
      7. Geohash encoding           (geohash_udf)
      8. 30-minute sliding-window aggregation per geohash
    """
    # ── Instantiate UDFs ──────────────────────────────────────────────────────
    is_relevant_udf   = make_tweet_is_relevant_udf()
    bot_score_udf     = make_tweet_bot_score_udf()
    clean_udf         = make_tweet_clean_udf()
    credibility_udf   = make_tweet_credibility_udf()
    disaster_type_udf = make_tweet_disaster_type_udf()
    geohash_udf       = make_geohash_udf(precision=5)

    MIN_CREDIBILITY = float(os.getenv("MIN_TWEET_CREDIBILITY", "0.30"))
    MAX_BOT_SCORE   = float(os.getenv("MAX_BOT_SCORE",         "0.70"))

    logger.info(f"[TweetStream] Preprocessing (min_cred={MIN_CREDIBILITY}, max_bot={MAX_BOT_SCORE})")

    # ── Step 1-2: Geo + relevance gate ────────────────────────────────────────
    filtered = (
        tweet_df
        .filter(F.col("lat").isNotNull() & F.col("lon").isNotNull())
        .filter(F.col("text").isNotNull() & (F.length(F.col("text")) > 5))
        .filter(is_relevant_udf(F.col("text")))          # disaster keyword gate
    )

    # ── Step 3: Text cleaning ─────────────────────────────────────────────────
    cleaned = filtered.withColumn("cleaned_text", clean_udf(F.col("text")))

    # ── Step 4: Bot score ─────────────────────────────────────────────────────
    bot_scored = (
        cleaned.withColumn(
            "bot_score",
            bot_score_udf(
                F.col("followers_count"),
                F.col("retweet_count"),
                F.col("account_age_days"),
                F.col("verified"),
            )
        )
        .filter(F.col("bot_score") < MAX_BOT_SCORE)     # filter likely bots
    )

    # ── Step 5: Credibility scoring ───────────────────────────────────────────
    cred_scored = (
        bot_scored.withColumn(
            "credibility_score",
            credibility_udf(
                F.col("text"),
                F.col("followers_count"),
                F.col("retweet_count"),
                F.col("verified"),
                F.col("account_age_days"),
                F.col("lat"),
                F.col("lon"),
            )
        )
        .filter(F.col("credibility_score") > MIN_CREDIBILITY)   # credibility gate
    )

    # ── Step 6-7: Type tagging + geohash ─────────────────────────────────────
    tagged = (
        cred_scored
        .withColumn("disaster_type_tweet", disaster_type_udf(F.col("text")))
        .withColumn("geohash",             geohash_udf(F.col("lat"), F.col("lon")))
        .withColumn("event_time",          F.to_timestamp("timestamp"))
        .withWatermark("event_time", "15 minutes")
    )

    # ── Step 8: 30-min sliding window aggregation per geohash ─────────────────
    aggregated = (
        tagged
        .groupBy(
            F.window("event_time", "30 minutes", "5 minutes"),
            F.col("geohash"),
        )
        .agg(
            F.count("*")                          .alias("tweet_count"),
            F.avg("credibility_score")            .alias("mean_credibility"),
            F.max("credibility_score")            .alias("max_credibility"),
            F.avg("bot_score")                    .alias("mean_bot_score"),
            F.avg("lat")                          .alias("centroid_lat"),
            F.avg("lon")                          .alias("centroid_lon"),
            F.collect_list("credibility_score")   .alias("cred_scores"),
            F.collect_list("cleaned_text")        .alias("cleaned_texts"),
            # Most common disaster type in this window
            F.first("disaster_type_tweet")        .alias("dominant_disaster_type"),
            # Source breakdown
            F.count(
                F.when(F.col("source") == "bluesky",  1)
            )                                     .alias("bluesky_count"),
            F.count(
                F.when(F.col("source") == "mastodon", 1)
            )                                     .alias("mastodon_count"),
            F.count(
                F.when(F.col("source") == "mock",     1)
            )                                     .alias("mock_count"),
        )
        # Minimum tweet volume to consider the window a signal
        .filter(F.col("tweet_count") >= 1)
    )

    return aggregated


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – SAR STREAM
# ═══════════════════════════════════════════════════════════════════════════════

def read_sar_stream(spark: SparkSession) -> DataFrame:
    """
    Read SAR file metadata from Kafka topic 'sar-raw'.
    The SAR image data itself is processed via SAR UDFs on executor nodes.
    """
    logger.info(f"[SARStream] Connecting to Kafka @ {KAFKA_BOOTSTRAP}, topic={TOPIC_SAR_RAW}")
    raw = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe",              TOPIC_SAR_RAW)
        .option("startingOffsets",        "latest")
        .option("maxOffsetsPerTrigger",   1_000)
        .option("failOnDataLoss",         "false")
        .load()
    )
    return (
        raw.select(
            F.from_json(F.col("value").cast("string"), SAR_SCHEMA).alias("d"),
            F.col("timestamp").alias("kafka_ts"),
        )
        .select("d.*", "kafka_ts")
    )


def preprocess_sar_stream(sar_df: DataFrame) -> DataFrame:
    """
    Apply the SAR preprocessing pipeline using UDFs that delegate to
    utils/sar_preprocessing.py:

      1. File quality gate    (sar_quality_udf)
      2. Label extraction     (sar_label_udf)
      3. SAR stats computation (sar_stats_udf) → Lee filter → dB → normalise
      4. Geohash from label/path (approximated — real SAR has embedded geo)
      5. Watermark + event time
    """
    quality_udf = make_sar_quality_udf()
    stats_udf   = make_sar_stats_udf()
    label_udf   = make_sar_label_udf()

    logger.info("[SARStream] Preprocessing SAR metadata with quality + stats UDFs")

    # ── Step 1: Quality gate ──────────────────────────────────────────────────
    valid = (
        sar_df
        .filter(F.col("filepath").isNotNull())
        .withColumn("sar_valid", quality_udf(F.col("filepath")))
        .filter(F.col("sar_valid") == True)
    )

    # ── Step 2: Label from path ───────────────────────────────────────────────
    labelled = valid.withColumn(
        "sar_label",
        F.coalesce(F.col("label"), label_udf(F.col("filepath")))
    )

    # ── Step 3: SAR statistics (runs Lee filter + dB conversion on executor) ──
    stats = labelled.withColumn(
        "sar_stats_json",
        stats_udf(F.col("filepath"))
    )

    # ── Step 4: Parse stats JSON into struct ──────────────────────────────────
    sar_stats_schema = StructType([
        StructField("valid",      BooleanType(), True),
        StructField("vv_mean",    FloatType(),   True),
        StructField("vv_std",     FloatType(),   True),
        StructField("vh_mean",    FloatType(),   True),
        StructField("vh_std",     FloatType(),   True),
        StructField("global_min", FloatType(),   True),
        StructField("global_max", FloatType(),   True),
    ])

    with_stats = stats.withColumn(
        "sar_stats",
        F.from_json(F.col("sar_stats_json"), sar_stats_schema)
    )

    # ── Step 5: Event time + watermark ───────────────────────────────────────
    processed = (
        with_stats
        .withColumn("event_time", F.to_timestamp("timestamp"))
        .withWatermark("event_time", "30 minutes")
        .select(
            F.col("filepath"),
            F.col("filename"),
            F.col("sar_label"),
            F.col("event_time"),
            F.col("kafka_ts"),
            F.col("sar_valid"),
            F.col("sar_stats_json"),
            F.col("sar_stats.vv_mean")    .alias("vv_mean"),
            F.col("sar_stats.vv_std")     .alias("vv_std"),
            F.col("sar_stats.vh_mean")    .alias("vh_mean"),
            F.col("sar_stats.vh_std")     .alias("vh_std"),
            F.col("sar_stats.global_min") .alias("sar_global_min"),
            F.col("sar_stats.global_max") .alias("sar_global_max"),
        )
    )

    return processed


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – FUSION: JOIN + INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def fuse_streams_and_infer(
    tweet_agg: DataFrame,
    sar_processed: DataFrame,
    spark: SparkSession,
) -> DataFrame:
    """
    Join the aggregated tweet window with SAR metadata in a 1-hour time window,
    then run the inference UDF to produce disaster alerts.

    Join strategy:
      • Stream-stream inner join on a 1-hour event-time window
      • If no SAR data is available in the window, fall back to tweet-only inference

    Because stream-stream joins require both sides to be streaming DataFrames
    with watermarks, we use a time-range condition:
        sar.event_time BETWEEN tweet.window.start - 1hr AND tweet.window.end + 1hr
    """
    inference_udf = make_inference_udf(spark, MODEL_PATH)
    logger.info(f"[Fusion] Using model: {MODEL_PATH}")

    # ── Stream-stream join ────────────────────────────────────────────────────
    # We join on a broad time window (SAR orbit repeat is 6-12 days, so for
    # demo/real-time we match within ±1 hour of the tweet window start).
    # In production you would join on sar_acquisition_timestamp ± window.
    fused = (
        tweet_agg.alias("tw")
        .join(
            sar_processed.alias("sar"),
            F.expr("""
                sar.event_time >= tw.window.start - interval 1 hour
                AND sar.event_time <= tw.window.end + interval 1 hour
            """),
            how="left",   # keep tweet windows even when no SAR is available
        )
        .select(
            F.col("tw.window"),
            F.col("tw.geohash"),
            F.col("tw.centroid_lat"),
            F.col("tw.centroid_lon"),
            F.col("tw.tweet_count"),
            F.col("tw.mean_credibility"),
            F.col("tw.max_credibility"),
            F.col("tw.mean_bot_score"),
            F.col("tw.cred_scores"),
            F.col("tw.dominant_disaster_type"),
            F.col("tw.bluesky_count"),
            F.col("tw.mastodon_count"),
            F.col("tw.mock_count"),
            # SAR side (may be null if no SAR in window)
            F.col("sar.filepath")        .alias("sar_filepath"),
            F.col("sar.sar_label")       .alias("sar_label"),
            F.col("sar.sar_stats_json"),
            F.col("sar.vv_mean"),
            F.col("sar.vv_std"),
            F.col("sar.vh_mean"),
            F.col("sar.vh_std"),
            F.col("sar.sar_global_min"),
            F.col("sar.sar_global_max"),
        )
    )

    # ── Run inference UDF ─────────────────────────────────────────────────────
    inferred = fused.withColumn(
        "inference_json",
        inference_udf(
            F.col("sar_stats_json"),                         # SAR stats (may be null)
            F.to_json(F.col("cred_scores")),                 # tweet credibility list as JSON
            F.col("tweet_count"),
            F.col("mean_credibility"),
        )
    )

    # ── Parse inference result ────────────────────────────────────────────────
    result = (
        inferred
        .withColumn("infer", F.from_json("inference_json", INFERENCE_RESULT_SCHEMA))
        .select(
            # Time & location
            F.col("window"),
            F.col("window.start")                            .alias("window_start"),
            F.col("window.end")                              .alias("window_end"),
            F.col("geohash"),
            F.col("centroid_lat"),
            F.col("centroid_lon"),
            # Tweet aggregates
            F.col("tweet_count"),
            F.col("mean_credibility"),
            F.col("max_credibility"),
            F.col("mean_bot_score"),
            F.col("dominant_disaster_type"),
            F.col("bluesky_count"),
            F.col("mastodon_count"),
            F.col("mock_count"),
            # SAR info
            F.col("sar_filepath"),
            F.col("sar_label"),
            F.col("vv_mean"),
            F.col("vv_std"),
            F.col("vh_mean"),
            F.col("vh_std"),
            # Inference output
            F.col("infer.disaster_type")                     .alias("disaster_type"),
            F.col("infer.severity")                          .alias("severity"),
            F.col("infer.confidence")                        .alias("confidence"),
            F.col("infer.is_disaster")                       .alias("is_disaster"),
            F.col("infer.source")                            .alias("inference_source"),
            # Processing timestamp
            F.current_timestamp()                            .alias("processed_at"),
        )
    )

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – SINKS
# ═══════════════════════════════════════════════════════════════════════════════

def write_alerts_to_kafka(alert_df: DataFrame,
                          query_name: str = "alerts_kafka") -> StreamingQuery:
    """Write detected alerts to the 'alerts' Kafka topic as JSON."""
    alert_payload = alert_df.select(
        F.to_json(F.struct(
            F.col("window_start")          .alias("window_start"),
            F.col("geohash"),
            F.col("centroid_lat")          .alias("lat"),
            F.col("centroid_lon")          .alias("lon"),
            F.col("tweet_count"),
            F.col("mean_credibility"),
            F.col("disaster_type"),
            F.col("severity"),
            F.col("confidence"),
            F.col("is_disaster"),
            F.col("dominant_disaster_type"),
            F.col("sar_label"),
            F.col("inference_source"),
            F.col("processed_at"),
        )).alias("value")
    )

    return (
        alert_payload.writeStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("topic",                   TOPIC_ALERTS)
        .option("checkpointLocation",      f"{CHECKPOINT_BASE}/{query_name}")
        .outputMode("append")
        .queryName(query_name)
        .start()
    )


def write_alerts_to_postgres(alert_df: DataFrame,
                             query_name: str = "alerts_postgres") -> StreamingQuery:
    """
    Write alerts to PostgreSQL (PostGIS) via JDBC foreachBatch.
    Creates the table on first run if it doesn't exist.
    """
    _db_url = DATABASE_URL

    def write_batch(batch_df: DataFrame, epoch_id: int):
        if batch_df.isEmpty():
            return
        try:
            (
                batch_df.select(
                    F.col("window_start")          .cast("timestamp").alias("timestamp"),
                    F.col("centroid_lat")          .alias("lat"),
                    F.col("centroid_lon")          .alias("lon"),
                    F.col("geohash"),
                    F.col("disaster_type"),
                    F.col("severity"),
                    F.col("confidence"),
                    F.col("is_disaster"),
                    F.col("tweet_count"),
                    F.col("mean_credibility"),
                    F.col("sar_label"),
                    F.col("inference_source"),
                    F.col("processed_at"),
                )
                .write
                .format("jdbc")
                .option("url",      _db_url.replace("postgresql://", "jdbc:postgresql://"))
                .option("dbtable",  "disaster_alerts_stream")
                .option("driver",   "org.postgresql.Driver")
                .mode("append")
                .save()
            )
            logger.info(f"[Postgres] Wrote batch {epoch_id} ({batch_df.count()} rows)")
        except Exception as exc:
            logger.error(f"[Postgres] Batch {epoch_id} write failed: {exc}")

    return (
        alert_df.writeStream
        .foreachBatch(write_batch)
        .option("checkpointLocation", f"{CHECKPOINT_BASE}/{query_name}")
        .outputMode("append")
        .queryName(query_name)
        .start()
    )


def write_all_events_to_console(alert_df: DataFrame,
                                query_name: str = "console_debug") -> StreamingQuery:
    """Debug sink: pretty-print every micro-batch to the console."""
    return (
        alert_df.select(
            F.col("window_start"),
            F.col("geohash"),
            F.col("tweet_count"),
            F.round(F.col("mean_credibility"), 3) .alias("mean_cred"),
            F.col("disaster_type"),
            F.col("severity"),
            F.round(F.col("confidence"), 3)       .alias("confidence"),
            F.col("is_disaster"),
            F.col("sar_label"),
            F.col("inference_source"),
        )
        .writeStream
        .outputMode("append")
        .format("console")
        .option("truncate",  "false")
        .option("numRows",   20)
        .queryName(query_name)
        .start()
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 – TWEET-ONLY FALLBACK PIPELINE
#   When no SAR data is available (e.g. during demo / tests without SAR data),
#   run inference on tweet aggregates alone without the stream-stream join.
# ═══════════════════════════════════════════════════════════════════════════════

def run_tweet_only_pipeline(tweet_agg: DataFrame,
                            spark: SparkSession) -> DataFrame:
    """
    Inference on tweet aggregates only (no SAR join).
    Used when SAR Kafka topic is not populated.
    """
    inference_udf = make_inference_udf(spark, MODEL_PATH)

    result = (
        tweet_agg
        .withColumn(
            "inference_json",
            inference_udf(
                F.lit(None),                             # no SAR stats
                F.to_json(F.col("cred_scores")),
                F.col("tweet_count"),
                F.col("mean_credibility"),
            )
        )
        .withColumn("infer", F.from_json("inference_json", INFERENCE_RESULT_SCHEMA))
        .select(
            F.col("window"),
            F.col("window.start")                        .alias("window_start"),
            F.col("window.end")                          .alias("window_end"),
            F.col("geohash"),
            F.col("centroid_lat"),
            F.col("centroid_lon"),
            F.col("tweet_count"),
            F.col("mean_credibility"),
            F.col("max_credibility"),
            F.col("mean_bot_score"),
            F.col("dominant_disaster_type"),
            F.col("bluesky_count"),
            F.col("mastodon_count"),
            F.col("mock_count"),
            F.lit(None).cast("string")                   .alias("sar_filepath"),
            F.lit(None).cast("string")                   .alias("sar_label"),
            F.lit(None).cast("float")                    .alias("vv_mean"),
            F.lit(None).cast("float")                    .alias("vv_std"),
            F.lit(None).cast("float")                    .alias("vh_mean"),
            F.lit(None).cast("float")                    .alias("vh_std"),
            F.col("infer.disaster_type")                 .alias("disaster_type"),
            F.col("infer.severity")                      .alias("severity"),
            F.col("infer.confidence")                    .alias("confidence"),
            F.col("infer.is_disaster")                   .alias("is_disaster"),
            F.col("infer.source")                        .alias("inference_source"),
            F.current_timestamp()                        .alias("processed_at"),
        )
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 – MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def run_streaming_pipeline(
    mode: str = "tweet_only",
    enable_kafka_sink: bool = True,
    enable_postgres_sink: bool = False,
    enable_console_sink: bool = True,
    await_termination: bool = True,
):
    """
    Launch the full Spark Structured Streaming pipeline.

    Args:
        mode: "tweet_only"  – tweet stream only (no SAR join) [default for demo]
              "fusion"      – full tweet + SAR stream-stream join + inference
        enable_kafka_sink:    Write alerts back to Kafka 'alerts' topic.
        enable_postgres_sink: Write alerts to PostgreSQL (needs PostGIS running).
        enable_console_sink:  Print micro-batches to stdout for debugging.
        await_termination:    Block until any query terminates (or Ctrl-C).
    """
    if not PYSPARK_AVAILABLE:
        logger.error("PySpark not available. Install: pip install pyspark>=3.4.0")
        return

    # ── 1. Spark session ──────────────────────────────────────────────────────
    local_mode = os.getenv("SPARK_LOCAL_MODE", "true").lower() == "true"
    spark = create_spark_session(
        app_name="DisasterDetection_Streaming",
        local_mode=local_mode,
    )
    logger.info(f"[Pipeline] Spark version: {spark.version}  |  mode={mode}  |  local={local_mode}")

    queries: list[StreamingQuery] = []

    # ── 2. Tweet stream ───────────────────────────────────────────────────────
    raw_tweets    = read_tweet_stream(spark)
    tweet_agg     = preprocess_tweet_stream(raw_tweets)
    logger.info("[Pipeline] Tweet stream preprocessing configured ✓")

    # ── 3. SAR stream (fusion mode only) ─────────────────────────────────────
    if mode == "fusion":
        raw_sar       = read_sar_stream(spark)
        sar_processed = preprocess_sar_stream(raw_sar)
        alert_df      = fuse_streams_and_infer(tweet_agg, sar_processed, spark)
        logger.info("[Pipeline] SAR stream preprocessing + fusion configured ✓")
    else:
        # Tweet-only: inference without SAR
        alert_df = run_tweet_only_pipeline(tweet_agg, spark)
        logger.info("[Pipeline] Tweet-only inference pipeline configured ✓")

    # ── 4. Filter to disaster-positive events ─────────────────────────────────
    disaster_alerts = alert_df.filter(F.col("is_disaster") == True)

    # ── 5. Sinks ──────────────────────────────────────────────────────────────
    if enable_console_sink:
        q = write_all_events_to_console(alert_df)        # all events (debug)
        queries.append(q)
        logger.info("[Pipeline] Console sink started ✓")

    if enable_kafka_sink:
        q = write_alerts_to_kafka(disaster_alerts)
        queries.append(q)
        logger.info(f"[Pipeline] Kafka sink started → topic={TOPIC_ALERTS} ✓")

    if enable_postgres_sink:
        q = write_alerts_to_postgres(disaster_alerts)
        queries.append(q)
        logger.info("[Pipeline] PostgreSQL sink started ✓")

    logger.info(
        f"[Pipeline] {len(queries)} streaming quer{'y' if len(queries)==1 else 'ies'} active. "
        f"Kafka={KAFKA_BOOTSTRAP}"
    )

    # ── 6. Await ──────────────────────────────────────────────────────────────
    if await_termination:
        try:
            spark.streams.awaitAnyTermination()
        except KeyboardInterrupt:
            logger.info("[Pipeline] Interrupted – stopping all queries...")
            for q in queries:
                try:
                    q.stop()
                except Exception:
                    pass
            spark.stop()
            logger.info("[Pipeline] Shutdown complete.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Disaster Detection – Spark Structured Streaming Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["tweet_only", "fusion"],
        default="tweet_only",
        help="'tweet_only': tweets only (default). 'fusion': SAR + tweets joined."
    )
    parser.add_argument("--no-kafka",    action="store_true", help="Disable Kafka alert sink")
    parser.add_argument("--postgres",    action="store_true", help="Enable PostgreSQL sink")
    parser.add_argument("--no-console",  action="store_true", help="Disable console sink")
    args = parser.parse_args()

    run_streaming_pipeline(
        mode=args.mode,
        enable_kafka_sink=not args.no_kafka,
        enable_postgres_sink=args.postgres,
        enable_console_sink=not args.no_console,
    )
