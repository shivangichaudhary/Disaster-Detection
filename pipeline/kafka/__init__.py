"""
pipeline/kafka/__init__.py
───────────────────────────
Public surface of the pipeline.kafka package.

Exports:
    KafkaProducerWrapper      – thread-safe JSON Kafka producer
    DisasterTwitterProducer   – filtered Twitter stream → Kafka
    SARImageProducer          – local/S3 SAR file watcher → Kafka
    AlertConsumer             – Kafka 'alerts' topic → PostGIS

Topics:
    sar-raw      raw SAR metadata (file path + acquisition info)
    tweets-raw   geo-tagged disaster tweets (pre-filtered)
    alerts       final detection results (binary + type + severity)
"""

from pipeline.kafka.producers import (
    KafkaProducerWrapper,
    DisasterTwitterProducer,
    SARImageProducer,
    AlertConsumer,
)

__all__ = [
    "KafkaProducerWrapper",
    "DisasterTwitterProducer",
    "SARImageProducer",
    "AlertConsumer",
]
