"""
pipeline/kafka/producer_twitter.py  +  producer_sar.py  +  consumer.py
───────────────────────────────────────────────────────────────────────
Real-time data ingestion via Apache Kafka.

Topics:
  sar-raw      → raw SAR image metadata + S3/local path
  tweets-raw   → filtered disaster tweets (geo + credibility)
  alerts       → final detection results for downstream consumers
"""

import os
import sys
import json
import time
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

try:
    from kafka import KafkaProducer, KafkaConsumer
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    logger.warning("kafka-python not installed. Using mock producer.")

try:
    import tweepy
    TWEEPY_AVAILABLE = True
except ImportError:
    TWEEPY_AVAILABLE = False

from dotenv import load_dotenv
load_dotenv()


KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")


# ─────────────────────────────────────────────────────────────────────────────
# Base Kafka wrapper
# ─────────────────────────────────────────────────────────────────────────────

class KafkaProducerWrapper:
    """Thread-safe Kafka producer with JSON serialisation and retry logic."""

    def __init__(self, bootstrap_servers: str = KAFKA_BOOTSTRAP, retries: int = 3):
        if not KAFKA_AVAILABLE:
            self._producer = None
            logger.warning("Using mock Kafka producer (kafka-python not installed).")
            return

        try:
            self._producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                retries=retries,
                acks="all",                     # wait for all replicas
                compression_type="gzip",
                max_request_size=10_485_760,    # 10 MB for SAR metadata
            )
            logger.info(f"Kafka producer connected to {bootstrap_servers}")
        except Exception as e:
            self._producer = None
            logger.warning(f"Failed to connect to Kafka at {bootstrap_servers}: {e}. Falling back to mock producer.")

    def send(self, topic: str, value: dict, key: Optional[str] = None):
        if self._producer is None:
            logger.debug(f"[MOCK] → {topic}: {json.dumps(value)[:120]}")
            return

        future = self._producer.send(topic, value=value, key=key)
        try:
            future.get(timeout=10)
        except KafkaError as e:
            logger.error(f"Kafka send failed: {e}")

    def flush(self):
        if self._producer:
            self._producer.flush()

    def close(self):
        if self._producer:
            self._producer.close()


# ─────────────────────────────────────────────────────────────────────────────
# Twitter/X Stream Producer
# ─────────────────────────────────────────────────────────────────────────────

DISASTER_FILTER_RULES = [
    # (value, tag) — Twitter API v2 filter rules
    ("(flood OR flooding OR floods) has:geo lang:en", "flood"),
    ("(earthquake OR quake OR tremor) has:geo lang:en", "earthquake"),
    ("(wildfire OR bushfire OR forest fire) has:geo lang:en", "wildfire"),
    ("(cyclone OR hurricane OR typhoon) has:geo lang:en", "cyclone"),
    ("(landslide OR mudslide) has:geo lang:en", "landslide"),
]


class DisasterTwitterProducer:
    """
    Connects to Twitter Filtered Stream API v2 and streams
    geo-tagged disaster tweets into Kafka topic 'tweets-raw'.

    Requires in .env:
        TWITTER_BEARER_TOKEN=...
    """

    TOPIC = "tweets-raw"

    def __init__(self, producer: KafkaProducerWrapper):
        self.producer    = producer
        self.bearer_token= os.getenv("TWITTER_BEARER_TOKEN")
        self.running     = False
        self._tweet_count= 0

    def _setup_rules(self, client: "tweepy.StreamingClient"):
        """Clear existing rules and add disaster filter rules."""
        existing = client.get_rules()
        if existing.data:
            ids = [rule.id for rule in existing.data]
            client.delete_rules(ids)
            logger.info(f"Cleared {len(ids)} existing stream rules.")

        rules = [tweepy.StreamRule(value=v, tag=t) for v, t in DISASTER_FILTER_RULES]
        result = client.add_rules(rules)
        logger.info(f"Added {len(result.data)} filter rules.")

    def start(self):
        """Start the Twitter stream. Blocks until stopped."""
        if not TWEEPY_AVAILABLE:
            logger.warning("tweepy not installed. Running mock tweet producer.")
            self._run_mock()
            return

        if not self.bearer_token:
            logger.warning("TWITTER_BEARER_TOKEN not set. Running mock tweet producer.")
            self._run_mock()
            return

        client = DisasterStreamClient(
            bearer_token=self.bearer_token,
            kafka_producer=self.producer,
            topic=self.TOPIC,
        )
        self._setup_rules(client)
        self.running = True

        logger.info("Starting Twitter disaster stream...")
        client.filter(
            tweet_fields=["created_at", "geo", "public_metrics", "author_id"],
            user_fields=["public_metrics", "verified", "created_at"],
            place_fields=["bounding_box", "country"],
            expansions=["author_id", "geo.place_id"],
        )

    def _run_mock(self, interval: float = 2.0):
        """Generate synthetic disaster tweets at regular intervals for testing."""
        import random
        templates = [
            ("Flooding in {city}! Roads underwater. #flood", 19.0, 72.8),
            ("Earthquake shaking {city}! Building damage. #earthquake", 28.6, 77.2),
            ("Wildfire spreading near {city}. #wildfire", 34.0, 74.0),
            ("Cyclone warning for {city}. #cyclone", 13.0, 80.2),
            ("Landslide blocks roads in {city}. #landslide", 27.5, 85.3),
        ]
        cities = ["Mumbai", "Delhi", "Chennai", "Kolkata", "Hyderabad", "Bangalore"]
        self.running = True
        logger.info("Mock tweet producer running...")

        while self.running:
            tmpl, lat, lon = random.choice(templates)
            city = random.choice(cities)
            tweet = {
                "text":              tmpl.format(city=city),
                "lat":               lat + random.uniform(-1, 1),
                "lon":               lon + random.uniform(-1, 1),
                "timestamp":         datetime.now(timezone.utc).isoformat(),
                "author_id":         str(random.randint(1000, 999999)),
                "followers_count":   random.randint(100, 50000),
                "retweet_count":     random.randint(0, 500),
                "verified":          False,
                "account_age_days":  random.randint(30, 2000),
                "source":            "mock",
            }
            self.producer.send(self.TOPIC, tweet, key=f"{tweet['lat']:.2f}_{tweet['lon']:.2f}")
            self._tweet_count += 1
            if self._tweet_count % 10 == 0:
                logger.debug(f"Produced {self._tweet_count} mock tweets")
            time.sleep(interval)

    def stop(self):
        self.running = False


if TWEEPY_AVAILABLE:
    class DisasterStreamClient(tweepy.StreamingClient):
        def __init__(self, bearer_token, kafka_producer, topic, **kwargs):
            super().__init__(bearer_token, **kwargs)
            self._kafka = kafka_producer
            self._topic = topic
            self._count = 0

        def on_tweet(self, tweet):
            try:
                data = {
                    "id":           tweet.id,
                    "text":         tweet.text,
                    "author_id":    str(tweet.author_id) if tweet.author_id else None,
                    "created_at":   tweet.created_at.isoformat() if tweet.created_at else None,
                    "timestamp":    datetime.now(timezone.utc).isoformat(),
                }
                # Extract geo
                if tweet.geo:
                    coords = tweet.geo.get("coordinates")
                    if coords:
                        data["lon"] = coords["coordinates"][0]
                        data["lat"] = coords["coordinates"][1]

                if "lat" not in data:
                    return   # skip unlocated tweets

                self._kafka.send(self._topic, data, key=f"{data['lat']:.2f}_{data['lon']:.2f}")
                self._count += 1
                if self._count % 100 == 0:
                    logger.info(f"Streamed {self._count} tweets")
            except Exception as e:
                logger.error(f"Error processing tweet: {e}")

        def on_errors(self, errors):
            logger.error(f"Stream errors: {errors}")
            return True  # don't disconnect


# ─────────────────────────────────────────────────────────────────────────────
# SAR Image Producer
# ─────────────────────────────────────────────────────────────────────────────

class SARImageProducer:
    """
    Polls for new Sentinel-1 acquisitions and sends metadata to
    Kafka topic 'sar-raw'. The actual image processing happens in Spark.

    In production: connects to ESA DHUS or AWS Open Data.
    In testing: scans a local directory for new .tif files.
    """

    TOPIC = "sar-raw"

    def __init__(self, producer: KafkaProducerWrapper, watch_dir: str = "data/raw/sentinel1"):
        self.producer  = producer
        self.watch_dir = Path(watch_dir)
        self.processed = set()
        self.running   = False

    def start(self, poll_interval: int = 60):
        """Poll for new SAR files every poll_interval seconds."""
        self.running = True
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"SAR producer watching: {self.watch_dir}")

        while self.running:
            self._scan_and_produce()
            time.sleep(poll_interval)

    def _scan_and_produce(self):
        tif_files = list(self.watch_dir.rglob("*.tif")) + \
                    list(self.watch_dir.rglob("*.SAFE"))

        for fpath in tif_files:
            if str(fpath) in self.processed:
                continue

            label = fpath.parent.name  # directory name = label
            metadata = {
                "filepath":    str(fpath),
                "filename":    fpath.name,
                "label":       label,
                "timestamp":   datetime.now(timezone.utc).isoformat(),
                "size_bytes":  fpath.stat().st_size if fpath.exists() else 0,
                "source":      "sentinel1",
            }
            self.producer.send(self.TOPIC, metadata, key=fpath.stem)
            self.processed.add(str(fpath))
            logger.debug(f"SAR producer: queued {fpath.name}")

    def stop(self):
        self.running = False


# ─────────────────────────────────────────────────────────────────────────────
# Alert Consumer (reads results, writes to PostGIS)
# ─────────────────────────────────────────────────────────────────────────────

class AlertConsumer:
    """
    Consumes from 'alerts' topic and persists to PostGIS database.
    Runs in a background thread.
    """

    TOPIC = "alerts"

    def __init__(self, db_url: Optional[str] = None):
        self.db_url  = db_url or os.getenv("DATABASE_URL", "postgresql://localhost:5432/disaster_db")
        self.running = False
        self._db     = None

    def _init_db(self):
        """Initialize PostGIS connection."""
        try:
            from sqlalchemy import create_engine, text
            from sqlalchemy.orm import sessionmaker
            engine = create_engine(self.db_url)
            self._Session = sessionmaker(engine)
            # Create table if not exists
            with engine.connect() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS disaster_alerts (
                        id           SERIAL PRIMARY KEY,
                        timestamp    TIMESTAMPTZ NOT NULL,
                        lat          FLOAT NOT NULL,
                        lon          FLOAT NOT NULL,
                        geom         GEOMETRY(Point, 4326),
                        disaster_type VARCHAR(32),
                        severity      VARCHAR(16),
                        confidence    FLOAT,
                        raw_json     JSONB,
                        created_at   TIMESTAMPTZ DEFAULT NOW()
                    );
                    CREATE INDEX IF NOT EXISTS idx_alerts_geom
                        ON disaster_alerts USING GIST(geom);
                    CREATE INDEX IF NOT EXISTS idx_alerts_ts
                        ON disaster_alerts(timestamp);
                """))
                conn.commit()
            logger.info("PostGIS initialized.")
        except Exception as e:
            logger.warning(f"PostGIS not available: {e}. Alerts will be logged only.")
            self._Session = None

    def start(self):
        """Start consuming alerts."""
        self._init_db()
        self.running = True

        if not KAFKA_AVAILABLE:
            logger.warning("Kafka not available. Alert consumer not started.")
            return

        consumer = KafkaConsumer(
            self.TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            group_id="alert-postprocessor",
            auto_offset_reset="latest",
            enable_auto_commit=True,
        )

        logger.info("Alert consumer started.")
        for message in consumer:
            if not self.running:
                break
            alert = message.value
            self._save_alert(alert)

    def _save_alert(self, alert: dict):
        """Persist alert to PostGIS and log it."""
        dtype    = alert.get("disaster_type", "unknown")
        severity = alert.get("severity", "unknown")
        conf     = alert.get("confidence", 0.0)
        lat      = alert.get("lat", 0.0)
        lon      = alert.get("lon", 0.0)

        logger.info(
            f"ALERT: {dtype.upper()} | severity={severity} | "
            f"confidence={conf:.2f} | lat={lat:.3f}, lon={lon:.3f}"
        )

        if self._Session is None:
            return

        try:
            from sqlalchemy import text
            with self._Session() as session:
                session.execute(text("""
                    INSERT INTO disaster_alerts
                    (timestamp, lat, lon, geom, disaster_type, severity, confidence, raw_json)
                    VALUES (
                        :ts, :lat, :lon,
                        ST_SetSRID(ST_MakePoint(:lon, :lat), 4326),
                        :dtype, :severity, :conf, :raw_json::jsonb
                    )
                """), {
                    "ts":       alert.get("timestamp", datetime.now(timezone.utc).isoformat()),
                    "lat":      lat, "lon": lon,
                    "dtype":    dtype, "severity": severity, "conf": conf,
                    "raw_json": json.dumps(alert),
                })
                session.commit()
        except Exception as e:
            logger.error(f"DB insert failed: {e}")

    def stop(self):
        self.running = False


# ─────────────────────────────────────────────────────────────────────────────
# Quick test runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Kafka producers/consumers")
    parser.add_argument("--component", choices=["twitter", "sar", "alert", "all"],
                        default="twitter")
    args = parser.parse_args()

    prod = KafkaProducerWrapper()

    if args.component in ("twitter", "all"):
        tp = DisasterTwitterProducer(prod)
        t  = threading.Thread(target=tp.start, daemon=True)
        t.start()

    if args.component in ("sar", "all"):
        sp = SARImageProducer(prod)
        t2 = threading.Thread(target=sp.start, kwargs={"poll_interval": 30}, daemon=True)
        t2.start()

    if args.component in ("alert", "all"):
        ac = AlertConsumer()
        t3 = threading.Thread(target=ac.start, daemon=True)
        t3.start()

    logger.info("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping...")
        prod.close()
