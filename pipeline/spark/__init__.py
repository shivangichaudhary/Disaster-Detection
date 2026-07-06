"""
pipeline/spark/__init__.py
──────────────────────────
Public API for the Spark streaming pipeline package.

Quick start:

    from pipeline.spark import create_spark_session, run_streaming_pipeline

    spark = create_spark_session(local_mode=True)
    run_streaming_pipeline(mode="tweet_only")
"""

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
    # Tweet UDFs
    make_tweet_clean_udf,
    make_tweet_credibility_udf,
    make_tweet_bot_score_udf,
    make_tweet_disaster_type_udf,
    make_tweet_is_relevant_udf,
    # Geo UDFs
    make_geohash_udf,
    # SAR UDFs
    make_sar_quality_udf,
    make_sar_stats_udf,
    make_sar_label_udf,
    # Inference UDF
    make_inference_udf,
    # Schemas
    TWEET_SCHEMA,
    SAR_SCHEMA,
    INFERENCE_RESULT_SCHEMA,
)

from pipeline.spark.streaming_job import (
    read_tweet_stream,
    preprocess_tweet_stream,
    read_sar_stream,
    preprocess_sar_stream,
    fuse_streams_and_infer,
    run_tweet_only_pipeline,
    write_alerts_to_kafka,
    write_alerts_to_postgres,
    write_all_events_to_console,
    run_streaming_pipeline,
)

__all__ = [
    # Config
    "create_spark_session",
    "KAFKA_BOOTSTRAP",
    "DATABASE_URL",
    "MODEL_PATH",
    "CHECKPOINT_BASE",
    "TOPIC_TWEETS_RAW",
    "TOPIC_SAR_RAW",
    "TOPIC_ALERTS",
    # UDFs
    "make_tweet_clean_udf",
    "make_tweet_credibility_udf",
    "make_tweet_bot_score_udf",
    "make_tweet_disaster_type_udf",
    "make_tweet_is_relevant_udf",
    "make_geohash_udf",
    "make_sar_quality_udf",
    "make_sar_stats_udf",
    "make_sar_label_udf",
    "make_inference_udf",
    "TWEET_SCHEMA",
    "SAR_SCHEMA",
    "INFERENCE_RESULT_SCHEMA",
    # Streaming job
    "read_tweet_stream",
    "preprocess_tweet_stream",
    "read_sar_stream",
    "preprocess_sar_stream",
    "fuse_streams_and_infer",
    "run_tweet_only_pipeline",
    "write_alerts_to_kafka",
    "write_alerts_to_postgres",
    "write_all_events_to_console",
    "run_streaming_pipeline",
]
