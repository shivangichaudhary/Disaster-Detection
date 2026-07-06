"""
pipeline/kafka/twitter_stream.py
─────────────────────────────────
Connects Twitter/X Filtered Stream API v2 to the disaster detection pipeline.

Modes:
  1. real + kafka   — Live tweets → preprocess → Kafka topic 'tweets-raw'
  2. real + direct  — Live tweets → preprocess → PostGIS DB directly (no Kafka)
  3. mock + kafka   — Synthetic tweets → Kafka (for Kafka testing without API key)
  4. mock + direct  — Synthetic tweets logged locally (fully offline demo)

Usage:
  # Real stream → Kafka (requires TWITTER_BEARER_TOKEN in .env)
  python pipeline/kafka/twitter_stream.py --mode real --output kafka

  # Real stream → direct DB (no Kafka needed)
  python pipeline/kafka/twitter_stream.py --mode real --output direct

  # Mock stream (no credentials needed — good for demos)
  python pipeline/kafka/twitter_stream.py --mode mock --output direct

Requirements in .env:
  TWITTER_BEARER_TOKEN=your_token_here
  KAFKA_BOOTSTRAP_SERVERS=localhost:9092   (only for --output kafka)
  DATABASE_URL=postgresql://...            (only for --output direct)
"""

import os
import sys
import json
import time
import math
import random
import signal
import threading
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ── Optional dependencies ─────────────────────────────────────────────────────

try:
    import tweepy
    TWEEPY_AVAILABLE = True
except ImportError:
    TWEEPY_AVAILABLE = False
    logger.warning("tweepy not installed — mock mode only. Run: pip install tweepy")

try:
    from kafka import KafkaProducer
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    logger.warning("kafka-python not installed — direct mode only.")

try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

# ── Config ────────────────────────────────────────────────────────────────────

BEARER_TOKEN        = os.getenv("TWITTER_BEARER_TOKEN", "")
KAFKA_BOOTSTRAP     = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
DATABASE_URL        = os.getenv("DATABASE_URL", "postgresql://disaster:disaster123@localhost:5432/disaster_db")
KAFKA_TOPIC_TWEETS  = "tweets-raw"

# Twitter API v2 filter rules — covers 5 disaster types + requires geo
DISASTER_FILTER_RULES = [
    ("(flood OR flooding OR flooded OR flash flood) has:geo lang:en",     "flood"),
    ("(earthquake OR quake OR tremor OR aftershock OR seismic) has:geo lang:en", "earthquake"),
    ("(wildfire OR bushfire OR forest fire OR blaze) has:geo lang:en",    "wildfire"),
    ("(cyclone OR hurricane OR typhoon OR tropical storm) has:geo lang:en","cyclone"),
    ("(landslide OR mudslide OR rockfall OR debris flow) has:geo lang:en","landslide"),
]

# Cities with known disaster risk — used for geo-inference fallback
CITY_GEO = {
    "mumbai": (19.076, 72.878), "delhi": (28.614, 77.209),
    "chennai": (13.083, 80.270), "kolkata": (22.573, 88.364),
    "hyderabad": (17.385, 78.487), "bangalore": (12.972, 77.595),
    "karachi": (24.861, 67.011), "lahore": (31.549, 74.343),
    "dhaka": (23.810, 90.413), "kathmandu": (27.717, 85.324),
    "jakarta": (-6.209, 106.846), "manila": (14.600, 120.984),
    "tokyo": (35.676, 139.650), "istanbul": (41.008, 28.978),
    "colombo": (6.927, 79.861), "bangkok": (13.756, 100.502),
}


# ── Preprocessing helper ──────────────────────────────────────────────────────

def _extract_geo(tweet_data: dict) -> tuple[Optional[float], Optional[float]]:
    """
    Extract (lat, lon) from a tweet dict.
    Priority: direct coords > place bbox centroid > city-name text inference.
    """
    lat = tweet_data.get("lat") or tweet_data.get("latitude")
    lon = tweet_data.get("lon") or tweet_data.get("longitude")
    if lat and lon:
        return float(lat), float(lon)

    # Try place bbox
    place = tweet_data.get("place", {})
    if isinstance(place, dict):
        bb = place.get("bounding_box", {}).get("coordinates")
        if bb:
            lons = [p[0] for p in bb[0]]
            lats = [p[1] for p in bb[0]]
            import statistics
            return statistics.mean(lats), statistics.mean(lons)

    # Infer from city mention in text
    text_lower = tweet_data.get("text", "").lower()
    for city, coords in CITY_GEO.items():
        if city in text_lower:
            return coords[0] + random.uniform(-0.05, 0.05), coords[1] + random.uniform(-0.05, 0.05)

    return None, None


def _preprocess_tweet(raw: dict) -> Optional[dict]:
    """
    Clean, score, and enrich a raw tweet dict.
    Returns None if tweet should be dropped (bot, low credibility, no geo).
    """
    from utils.tweet_preprocessing import TweetCleaner, CredibilityScorer, BotDetector, to_geohash

    text = raw.get("text", "").strip()
    if not text or len(text) < 10:
        return None

    cleaner = TweetCleaner()
    cleaned = cleaner.clean(text)

    lat, lon = _extract_geo(raw)
    if lat is None:
        return None  # must be geolocated

    # Bot check
    bot_det = BotDetector()
    if bot_det.is_bot(raw, threshold=0.7):
        logger.debug("Dropped bot tweet")
        return None

    # Credibility score
    raw["lat"], raw["lon"] = lat, lon
    scorer = CredibilityScorer()
    cred = scorer.score(raw)

    # Disaster type
    dtype = cleaner.get_disaster_type(text) or "none"

    return {
        "id":               raw.get("id") or raw.get("tweet_id", ""),
        "text":             cleaned,
        "raw_text":         text,
        "lat":              lat,
        "lon":              lon,
        "geohash":          to_geohash(lat, lon, precision=5),
        "timestamp":        raw.get("created_at") or raw.get("timestamp") or datetime.now(timezone.utc).isoformat(),
        "author_id":        str(raw.get("author_id", "")),
        "followers_count":  int(raw.get("followers_count", 0)),
        "retweet_count":    int(raw.get("retweet_count", 0)),
        "like_count":       int(raw.get("like_count", 0)),
        "verified":         bool(raw.get("verified", False)),
        "account_age_days": int(raw.get("account_age_days", 30)),
        "credibility_score":cred,
        "disaster_type":    dtype,
        "is_disaster":      dtype != "none",
        "source":           raw.get("source", "twitter_api"),
    }


# ── Output backends ───────────────────────────────────────────────────────────

class KafkaOutput:
    """Sends processed tweets to Kafka topic 'tweets-raw'."""

    def __init__(self, bootstrap_servers: str = KAFKA_BOOTSTRAP):
        if not KAFKA_AVAILABLE:
            raise RuntimeError("kafka-python not installed.")
        self._producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            retries=3,
            acks="all",
            compression_type="gzip",
        )
        logger.info(f"Kafka output connected to {bootstrap_servers}")

    def send(self, tweet: dict):
        key = f"{tweet['lat']:.3f}_{tweet['lon']:.3f}"
        self._producer.send(KAFKA_TOPIC_TWEETS, value=tweet, key=key)

    def close(self):
        self._producer.flush()
        self._producer.close()


class DirectDBOutput:
    """
    Writes processed tweets directly to PostGIS disaster_alerts table.
    Use when Kafka is not available (development / lightweight deployment).
    """

    def __init__(self, db_url: str = DATABASE_URL):
        if not SQLALCHEMY_AVAILABLE:
            raise RuntimeError("SQLAlchemy not installed.")
        engine = create_engine(db_url)
        # Quick connection test
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        self._Session = sessionmaker(bind=engine)
        logger.info(f"Direct DB output connected: {engine.url.database}@{engine.url.host}")

    def send(self, tweet: dict):
        try:
            with self._Session() as session:
                session.execute(text("""
                    INSERT INTO disaster_alerts
                    (timestamp, lat, lon, geom, disaster_type, severity, confidence,
                     is_disaster, tweet_count, geohash, raw_json)
                    VALUES (
                        :ts, :lat, :lon,
                        ST_SetSRID(ST_MakePoint(:lon, :lat), 4326),
                        :dtype, 'medium', :cred, :is_disaster,
                        1, :geohash, :raw_json::jsonb
                    )
                """), {
                    "ts":         tweet["timestamp"],
                    "lat":        tweet["lat"],
                    "lon":        tweet["lon"],
                    "dtype":      tweet["disaster_type"],
                    "cred":       tweet["credibility_score"],
                    "is_disaster":tweet["is_disaster"],
                    "geohash":    tweet["geohash"],
                    "raw_json":   json.dumps(tweet),
                })
                session.commit()
        except Exception as e:
            logger.error(f"DB insert failed: {e}")

    def close(self):
        pass


class LogOutput:
    """Fallback — just prints tweets to the console. No external dependencies."""

    def __init__(self):
        logger.info("Log output mode (console only — no Kafka or DB required)")
        self._count = 0

    def send(self, tweet: dict):
        self._count += 1
        logger.info(
            f"[{self._count:>5}] {tweet['disaster_type'].upper():12} | "
            f"cred={tweet['credibility_score']:.2f} | "
            f"({tweet['lat']:.3f},{tweet['lon']:.3f}) | "
            f"{tweet['text'][:80]}..."
        )

    def close(self):
        logger.info(f"Log output closed. Total tweets: {self._count}")


# ── Real Twitter Stream (API v2) ──────────────────────────────────────────────

if TWEEPY_AVAILABLE:
    class _DisasterStreamClient(tweepy.StreamingClient):
        """
        Tweepy v2 Filtered Stream client.
        Handles geo extraction, bot filtering, credibility scoring,
        and routing to the configured output backend.
        """

        def __init__(self, bearer_token: str, output_backend, **kwargs):
            super().__init__(bearer_token, wait_on_rate_limit=True, **kwargs)
            self._output   = output_backend
            self._count    = 0
            self._dropped  = 0
            self._includes = {}

        def on_tweet(self, tweet):
            try:
                raw = {
                    "id":         str(tweet.id),
                    "text":       tweet.text,
                    "author_id":  str(tweet.author_id) if tweet.author_id else "",
                    "created_at": tweet.created_at.isoformat() if tweet.created_at else None,
                    "timestamp":  datetime.now(timezone.utc).isoformat(),
                    "source":     "twitter_api_v2",
                }

                # Geo from direct coordinates
                if tweet.geo:
                    geo = tweet.geo
                    if isinstance(geo, dict):
                        coords = geo.get("coordinates", {})
                        if coords and "coordinates" in coords:
                            raw["lon"] = coords["coordinates"][0]
                            raw["lat"] = coords["coordinates"][1]

                # Merge in user metadata from expansions (injected by on_includes)
                user_meta = self._includes.get(str(tweet.author_id), {})
                raw.update(user_meta)

                processed = _preprocess_tweet(raw)
                if processed:
                    self._output.send(processed)
                    self._count += 1
                    if self._count % 50 == 0:
                        logger.info(
                            f"Stream stats: {self._count} accepted, "
                            f"{self._dropped} dropped, "
                            f"{self._count + self._dropped} total"
                        )
                else:
                    self._dropped += 1

            except Exception as e:
                logger.error(f"Error processing tweet {tweet.id}: {e}")

        def on_includes(self, includes):
            """Cache user expansions for enriching tweet metadata."""
            users = includes.get("users", [])
                for user in users:
                    self._includes[str(user.id)] = {
                        "followers_count":  user.public_metrics.get("followers_count", 0)
                                            if user.public_metrics else 0,
                        "verified":         user.verified or False,
                        "account_age_days": (datetime.now(timezone.utc) - user.created_at).days
                                            if user.created_at else 30,
                    }

        def on_errors(self, errors):
            logger.error(f"Stream error: {errors}")
            return True  # keep running


class RealTwitterStream:
    """
    Manages the Twitter API v2 Filtered Stream with:
    - Automatic filter rule setup
    - Exponential backoff reconnection
    - Graceful shutdown on SIGINT/SIGTERM
    """

    MAX_RETRIES = 8
    BASE_BACKOFF = 5   # seconds

    def __init__(self, bearer_token: str, output_backend):
        self._token   = bearer_token
        self._output  = output_backend
        self._running = False
        self._client  = None

    def _setup_rules(self, client):
        """Wipe old rules and install the 5 disaster filter rules."""
        existing = client.get_rules()
        if existing.data:
            ids = [r.id for r in existing.data]
            client.delete_rules(ids)
            logger.info(f"Removed {len(ids)} old stream rules.")

        new_rules = [tweepy.StreamRule(value=v, tag=t) for v, t in DISASTER_FILTER_RULES]
        result = client.add_rules(new_rules)
        if result.errors:
            logger.error(f"Rule errors: {result.errors}")
        else:
            logger.info(f"Installed {len(result.data)} disaster filter rules:")
            for rule in result.data:
                logger.info(f"  [{rule.tag}] {rule.value}")

    def start(self):
        """Start streaming with exponential backoff on connection failures."""
        if not TWEEPY_AVAILABLE:
            raise RuntimeError("tweepy is required for real stream mode.")

        self._running = True
        retries = 0

        while self._running and retries <= self.MAX_RETRIES:
            try:
                self._client = _DisasterStreamClient(self._token, self._output)
                self._setup_rules(self._client)

                logger.info("▶ Twitter/X disaster stream started")
                self._client.filter(
                    tweet_fields=["created_at", "geo", "public_metrics", "author_id", "entities"],
                    user_fields=["public_metrics", "verified", "created_at"],
                    place_fields=["bounding_box", "country", "full_name"],
                    expansions=["author_id", "geo.place_id"],
                )

            except tweepy.errors.TwitterServerError as e:
                if not self._running:
                    break
                backoff = self.BASE_BACKOFF * (2 ** retries) + random.uniform(0, 2)
                logger.warning(f"Twitter server error: {e}. Reconnecting in {backoff:.1f}s...")
                time.sleep(backoff)
                retries += 1

            except Exception as e:
                if not self._running:
                    break
                backoff = self.BASE_BACKOFF * (2 ** min(retries, 6))
                logger.error(f"Stream error: {e}. Retrying in {backoff:.0f}s...")
                time.sleep(backoff)
                retries += 1

        if retries > self.MAX_RETRIES:
            logger.error(f"Max retries ({self.MAX_RETRIES}) reached. Giving up.")

    def stop(self):
        self._running = False
        if self._client:
            self._client.disconnect()
        logger.info("Twitter stream stopped.")


# ── Mock Stream ───────────────────────────────────────────────────────────────

class MockTwitterStream:
    """
    Generates realistic synthetic disaster tweets at configurable rate.
    Fully offline — no API credentials required.
    Great for demos, CI testing, and pipeline validation.
    """

    TEMPLATES = {
        "flood": [
            "Severe flooding in {city}! Roads completely underwater. #flood #SOS",
            "Flash floods hit {city} area. Emergency services deployed. #flooding",
            "Water levels rising dangerously near {city}. Evacuations underway. #flood",
            "BREAKING: Catastrophic flooding in {city}. Please avoid low-lying areas. #FloodAlert",
            "Homes submerged in {city} district. Rescue operations ongoing. #flood",
        ],
        "earthquake": [
            "Strong earthquake just hit {city}! Buildings shaking badly. #earthquake",
            "Magnitude 6.2 tremor felt in {city}. Multiple aftershocks reported. #quake",
            "EARTHQUAKE in {city} region. Please stay away from buildings. #tremor",
            "Rescue teams deployed to {city} after major earthquake. #earthquake",
            "Building collapse feared in {city} after 7.0 magnitude quake. #earthquake",
        ],
        "wildfire": [
            "Massive wildfire raging near {city}! Thousands evacuating. #wildfire",
            "Fire spreading rapidly towards {city} residential areas. #WildfireAlert",
            "Air quality critical in {city} as wildfire smoke blankets region. #wildfire",
            "Firefighters battling blaze near {city}. Evacuation orders issued. #fire",
            "Wildfire destroys hundreds of homes near {city}. #WildfireEmergency",
        ],
        "cyclone": [
            "Category 4 cyclone making landfall near {city}! Storm surge imminent. #cyclone",
            "Cyclone warning: extreme winds and heavy rain battering {city}. #CycloneAlert",
            "Typhoon approaching {city} coast. Port authorities closed. #typhoon",
            "Hurricane force winds reported in {city}. Seek shelter immediately. #hurricane",
            "Cyclone damage reports flooding in from {city} area. #disaster",
        ],
        "landslide": [
            "Massive landslide blocks main highway near {city}. #landslide",
            "Mudslide triggered by heavy rains traps residents near {city}. #mudslide",
            "Landslide buries homes in hillside community near {city}. #disaster",
            "Debris flow cuts off {city} from surrounding areas. #landslide",
            "Rockfall on mountain road near {city}. Search and rescue deployed. #rockfall",
        ],
        "none": [
            "Beautiful sunny day in {city} today! Perfect weather. #weather",
            "Heavy traffic on the expressway near {city} this morning. #traffic",
            "Cricket match in {city} — great atmosphere at the stadium! #cricket",
            "New metro line opening in {city} next month. #infrastructure",
            "Street food festival in {city} this weekend. Must visit! #food",
        ],
    }

    def __init__(self, output_backend, interval: float = 3.0, disaster_ratio: float = 0.8):
        self._output   = output_backend
        self._interval = interval
        self._ratio    = disaster_ratio   # fraction of tweets that are disaster-related
        self._running  = False
        self._count    = 0

    def start(self):
        self._running = True
        logger.info(f"▶ Mock disaster tweet stream started (interval={self._interval}s, "
                    f"disaster_ratio={self._ratio:.0%})")

        disaster_types  = ["flood", "earthquake", "wildfire", "cyclone", "landslide"]
        all_cities      = list(CITY_GEO.keys())

        while self._running:
            city  = random.choice(all_cities)
            lat0, lon0 = CITY_GEO[city]

            if random.random() < self._ratio:
                dtype = random.choice(disaster_types)
            else:
                dtype = "none"

            text = random.choice(self.TEMPLATES[dtype]).format(city=city.capitalize())

            raw = {
                "text":             text,
                "lat":              lat0 + random.uniform(-0.3, 0.3),
                "lon":              lon0 + random.uniform(-0.3, 0.3),
                "created_at":       datetime.now(timezone.utc).isoformat(),
                "author_id":        str(random.randint(10000, 9999999)),
                "followers_count":  random.randint(50, 200000),
                "retweet_count":    random.randint(0, 2000),
                "like_count":       random.randint(0, 5000),
                "verified":         random.random() > 0.95,
                "account_age_days": random.randint(30, 3000),
                "source":           "mock",
            }

            processed = _preprocess_tweet(raw)
            if processed:
                self._output.send(processed)
                self._count += 1

            time.sleep(self._interval)

    def stop(self):
        self._running = False
        logger.info(f"Mock stream stopped. Generated {self._count} tweets.")


# ── Connection factory ────────────────────────────────────────────────────────

def build_output(output_mode: str):
    """Build the output backend based on user choice."""
    if output_mode == "kafka":
        try:
            return KafkaOutput(KAFKA_BOOTSTRAP)
        except Exception as e:
            logger.error(f"Kafka unavailable: {e}. Falling back to log output.")
            return LogOutput()

    elif output_mode == "direct":
        try:
            return DirectDBOutput(DATABASE_URL)
        except Exception as e:
            logger.warning(f"DB unavailable: {e}. Falling back to log output.")
            return LogOutput()

    else:
        return LogOutput()


def build_stream(stream_mode: str, output_backend):
    """Build the tweet stream based on user choice."""
    if stream_mode == "real":
        if not BEARER_TOKEN:
            logger.warning(
                "TWITTER_BEARER_TOKEN not set in .env — switching to mock mode.\n"
                "To use the real API:\n"
                "  1. Register at https://developer.twitter.com\n"
                "  2. Create a project with Filtered Stream access\n"
                "  3. Add TWITTER_BEARER_TOKEN=<your_token> to .env"
            )
            return MockTwitterStream(output_backend)
        return RealTwitterStream(BEARER_TOKEN, output_backend)

    return MockTwitterStream(output_backend)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Twitter/X → Disaster Detection stream connector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Real Twitter API → PostGIS DB (no Kafka)
  python pipeline/kafka/twitter_stream.py --mode real --output direct

  # Mock tweets → Kafka (pipeline testing without API credentials)
  python pipeline/kafka/twitter_stream.py --mode mock --output kafka

  # Fully offline demo (just prints to console)
  python pipeline/kafka/twitter_stream.py --mode mock --output log --interval 1.0
        """
    )
    parser.add_argument("--mode",     choices=["real", "mock"], default="mock",
                        help="'real' = Twitter API v2, 'mock' = synthetic generator")
    parser.add_argument("--output",   choices=["kafka", "direct", "log"], default="log",
                        help="'kafka' = Kafka topic, 'direct' = PostGIS, 'log' = console")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="Seconds between mock tweets (mock mode only)")
    parser.add_argument("--disaster-ratio", type=float, default=0.8,
                        help="Fraction of mock tweets that are disaster-related (0–1)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  Disaster Detection — Twitter/X Stream Connector")
    logger.info(f"  Stream mode : {args.mode.upper()}")
    logger.info(f"  Output mode : {args.output.upper()}")
    logger.info("=" * 60)

    output  = build_output(args.output)
    stream  = build_stream(args.mode, output)
    if isinstance(stream, MockTwitterStream):
        stream._interval = args.interval
        stream._ratio    = args.disaster_ratio

    # Graceful shutdown on Ctrl+C / SIGTERM
    def _shutdown(sig, frame):
        logger.info("Shutdown signal received...")
        stream.stop()
        output.close()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    stream.start()
    output.close()


if __name__ == "__main__":
    main()
