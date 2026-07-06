"""
pipeline/__init__.py
─────────────────────
Top-level pipeline package.

Sub-packages:
    pipeline.kafka  – Kafka producers and consumers
    pipeline.spark  – Spark Structured Streaming job and UDFs
"""

from pipeline.kafka import (
    KafkaProducerWrapper,
    DisasterTwitterProducer,
    SARImageProducer,
    AlertConsumer,
)

from pipeline.spark import (
    create_spark_session,
    KAFKA_BOOTSTRAP,
    DATABASE_URL,
    MODEL_PATH,
    CHECKPOINT_BASE,
    TOPIC_TWEETS_RAW,
    TOPIC_SAR_RAW,
    TOPIC_ALERTS,
    run_streaming_pipeline,
)

__all__ = [
    # Kafka
    "KafkaProducerWrapper",
    "DisasterTwitterProducer",
    "SARImageProducer",
    "AlertConsumer",
    # Spark config
    "create_spark_session",
    "KAFKA_BOOTSTRAP",
    "DATABASE_URL",
    "MODEL_PATH",
    "CHECKPOINT_BASE",
    "TOPIC_TWEETS_RAW",
    "TOPIC_SAR_RAW",
    "TOPIC_ALERTS",
    # Streaming
    "run_streaming_pipeline",
]
