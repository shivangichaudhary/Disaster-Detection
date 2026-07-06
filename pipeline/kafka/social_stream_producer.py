"""
pipeline/kafka/social_stream_producer.py
─────────────────────────────────────────
Real-time social media stream → Kafka topic 'tweets-raw'

Supports three sources (pick based on budget):
  1. Bluesky Jetstream   — FREE, no account needed, recommended
  2. Mastodon            — FREE, open-source, federated
  3. Twitter/X API v2    — $5000/month Pro tier (reference only)

Usage:
  python pipeline/kafka/social_stream_producer.py --source bluesky
  python pipeline/kafka/social_stream_producer.py --source mastodon
  python pipeline/kafka/social_stream_producer.py --source twitter   # needs paid API key
  python pipeline/kafka/social_stream_producer.py --source mock      # synthetic, always works
"""

import os
import re
import sys
import json
import time
import random
import asyncio
import argparse
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from loguru import logger
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pipeline.kafka.producers import KafkaProducerWrapper

KAFKA_TOPIC = "tweets-raw"

# ── Disaster keyword filter ───────────────────────────────────────────────────
DISASTER_KEYWORDS = [
    "flood", "flooding", "floods", "flash flood", "inundation",
    "earthquake", "quake", "tremor", "aftershock", "seismic",
    "wildfire", "forest fire", "bushfire", "blaze",
    "cyclone", "hurricane", "typhoon", "storm surge",
    "landslide", "mudslide", "avalanche",
    "disaster", "emergency", "evacuation", "SOS", "rescue",
    "#flood", "#earthquake", "#wildfire", "#cyclone", "#disaster",
]

def is_disaster_relevant(text: str) -> bool:
    t = text.lower()
    return any(kw.lower() in t for kw in DISASTER_KEYWORDS)


# ─────────────────────────────────────────────────────────────────────────────
# 1. BLUESKY JETSTREAM PRODUCER  (FREE — Recommended)
# ─────────────────────────────────────────────────────────────────────────────

JETSTREAM_URL = "wss://jetstream2.us-east.bsky.network/subscribe?wantedCollections=app.bsky.feed.post"

class BlueskyJetstreamProducer:
    """
    Connects to Bluesky's Jetstream WebSocket API.
    Streams all public posts in real time — completely free, no account needed.

    Jetstream delivers JSON events directly (no CBOR decoding needed).
    Docs: https://docs.bsky.app/blog/jetstream

    Posts are filtered for disaster keywords before sending to Kafka.
    Location is extracted from post text using regex (Bluesky has limited
    native geo-tagging, so we do NLP-based location extraction).
    """

    def __init__(self, kafka_producer: KafkaProducerWrapper):
        self.kafka    = kafka_producer
        self.running  = False
        self._count   = 0
        self._sent    = 0

    async def _stream(self):
        try:
            import websockets
        except ImportError:
            logger.error("websockets not installed. Run: pip install websockets")
            return

        logger.info(f"Connecting to Bluesky Jetstream: {JETSTREAM_URL}")

        while self.running:
            try:
                async with websockets.connect(
                    JETSTREAM_URL,
                    ping_interval=20,
                    ping_timeout=10,
                ) as ws:
                    logger.success("Connected to Bluesky Jetstream. Streaming posts...")
                    async for raw_msg in ws:
                        if not self.running:
                            break
                        try:
                            event = json.loads(raw_msg)
                            self._process_event(event)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                if self.running:
                    logger.warning(f"Jetstream connection lost: {e}. Reconnecting in 5s...")
                    await asyncio.sleep(5)

    def _process_event(self, event: dict):
        """Parse a Jetstream event and push disaster posts to Kafka."""
        self._count += 1

        # Only process new posts (not likes, follows, reposts)
        if event.get("kind") != "commit":
            return
        commit = event.get("commit", {})
        if commit.get("collection") != "app.bsky.feed.post":
            return
        if commit.get("operation") not in ("create",):
            return

        record = commit.get("record", {})
        text   = record.get("text", "")

        if not text or not is_disaster_relevant(text):
            return

        # Extract language
        langs = record.get("langs", ["en"])
        if langs and "en" not in langs:
            return  # English only for now

        # Extract location from text using simple regex
        lat, lon = self._extract_location_from_text(text)

        # Build standardised message (same schema as Twitter producer)
        message = {
            "text":             text,
            "lat":              lat,
            "lon":              lon,
            "timestamp":        event.get("time_us", "") or datetime.now(timezone.utc).isoformat(),
            "author_id":        event.get("did", ""),
            "followers_count":  0,        # not available from Jetstream
            "retweet_count":    0,
            "verified":         False,
            "account_age_days": 365,      # conservative default
            "source":           "bluesky",
            "platform":         "bluesky",
            "has_geo":          lat is not None,
        }

        # Use geohash as Kafka key for spatial locality
        key = f"{lat:.2f}_{lon:.2f}" if lat and lon else "no_geo"
        self.kafka.send(KAFKA_TOPIC, message, key=key)
        self._sent += 1

        if self._sent % 50 == 0:
            logger.info(f"Bluesky: processed {self._count} posts, sent {self._sent} disaster alerts to Kafka")

    def _extract_location_from_text(self, text: str):
        """
        Extract approximate lat/lon from post text.
        Simple approach: match known city names.
        For production: use spaCy NER + geocoder.
        """
        CITY_COORDS = {
            "mumbai":     (19.076, 72.877), "delhi":      (28.613, 77.209),
            "chennai":    (13.082, 80.270), "kolkata":    (22.572, 88.363),
            "bangalore":  (12.971, 77.594), "hyderabad":  (17.385, 78.486),
            "pakistan":   (30.375, 69.345), "lahore":     (31.558, 74.351),
            "karachi":    (24.860, 67.010), "dhaka":      (23.685, 90.356),
            "nepal":      (28.394, 84.124), "kathmandu":  (27.700, 85.318),
            "bangladesh": (23.685, 90.356), "sri lanka":  (7.873,  80.771),
            "turkey":     (38.963, 35.243), "istanbul":   (41.015, 28.979),
            "japan":      (36.204, 138.25), "tokyo":      (35.682, 139.691),
            "indonesia":  (-0.789, 113.921),"jakarta":    (-6.208, 106.845),
            "california": (36.778, -119.41),"los angeles":(34.052, -118.243),
            "florida":    (27.664, -81.515),"texas":      (31.968, -99.901),
            "india":      (20.593, 78.962), "china":      (35.861, 104.195),
            "australia":  (-25.27, 133.775),"sydney":     (-33.86, 151.209),
            "europe":     (54.526, 15.255),
        }
        text_lower = text.lower()
        for city, coords in CITY_COORDS.items():
            if city in text_lower:
                # Add small random offset so multiple posts from same city
                # don't all land on exactly the same point
                lat = coords[0] + random.uniform(-0.5, 0.5)
                lon = coords[1] + random.uniform(-0.5, 0.5)
                return round(lat, 4), round(lon, 4)
        return None, None

    def start(self):
        """Start streaming in the current thread (blocking)."""
        self.running = True
        asyncio.run(self._stream())

    def start_background(self) -> threading.Thread:
        """Start streaming in a background thread (non-blocking)."""
        self.running = True
        t = threading.Thread(target=self.start, daemon=True)
        t.start()
        return t

    def stop(self):
        self.running = False


# ─────────────────────────────────────────────────────────────────────────────
# 2. MASTODON PRODUCER  (FREE)
# ─────────────────────────────────────────────────────────────────────────────

class MastodonStreamProducer:
    """
    Streams public posts from Mastodon's public timeline via Server-Sent Events.
    Uses mastodon.social (largest instance) or any other instance.
    Completely free, no authentication needed for public timeline.

    Docs: https://docs.joinmastodon.org/methods/streaming/
    """

    # Public streaming endpoint — no auth needed
    STREAM_URL = "https://mastodon.social/api/v1/streaming/public"

    def __init__(self, kafka_producer: KafkaProducerWrapper,
                 instance: str = "mastodon.social"):
        self.kafka    = kafka_producer
        self.instance = instance
        self.running  = False
        self._count   = 0
        self._sent    = 0

    def start(self):
        """Stream Mastodon public timeline via SSE."""
        import requests

        url = f"https://{self.instance}/api/v1/streaming/public"
        logger.info(f"Connecting to Mastodon stream: {url}")
        self.running = True

        while self.running:
            try:
                with requests.get(url, stream=True, timeout=30,
                                  headers={"Accept": "text/event-stream"}) as resp:
                    logger.success(f"Connected to Mastodon ({self.instance})")
                    event_type = None
                    for line in resp.iter_lines(decode_unicode=True):
                        if not self.running:
                            break
                        if not line:
                            event_type = None
                            continue
                        if line.startswith("event:"):
                            event_type = line[6:].strip()
                        elif line.startswith("data:") and event_type == "update":
                            try:
                                data = json.loads(line[5:])
                                self._process_status(data)
                            except Exception:
                                pass
            except Exception as e:
                if self.running:
                    logger.warning(f"Mastodon stream error: {e}. Retrying in 5s...")
                    time.sleep(5)

    def _process_status(self, status: dict):
        """Process a Mastodon status (post)."""
        self._count += 1

        # Clean HTML tags from content
        content = status.get("content", "")
        text    = re.sub(r"<[^>]+>", "", content).strip()

        if not text or not is_disaster_relevant(text):
            return

        # Language filter
        if status.get("language") not in ("en", None):
            return

        # Approximate location
        account = status.get("account", {})
        lat, lon = None, None

        # Try to geocode from account location field
        acct_loc = account.get("source", {}).get("fields", [])
        for field in acct_loc:
            if "location" in field.get("name", "").lower():
                # Would use geocoder here; use city lookup for now
                lat, lon = self._city_lookup(field.get("value", ""))

        if lat is None:
            lat, lon = self._city_lookup(text)
        if lat is None:
            return  # skip posts with no location signal

        message = {
            "text":             text,
            "lat":              lat,
            "lon":              lon,
            "timestamp":        status.get("created_at", datetime.now(timezone.utc).isoformat()),
            "author_id":        account.get("id", ""),
            "followers_count":  account.get("followers_count", 0),
            "retweet_count":    status.get("reblogs_count", 0),
            "verified":         False,
            "account_age_days": 365,
            "source":           "mastodon",
            "platform":         "mastodon",
        }

        key = f"{lat:.2f}_{lon:.2f}"
        self.kafka.send(KAFKA_TOPIC, message, key=key)
        self._sent += 1

        if self._sent % 20 == 0:
            logger.info(f"Mastodon: processed {self._count}, sent {self._sent} to Kafka")

    def _city_lookup(self, text: str):
        """Same city lookup as Bluesky producer."""
        CITY_COORDS = {
            "mumbai": (19.076, 72.877), "delhi": (28.613, 77.209),
            "pakistan": (30.375, 69.345), "earthquake": (0, 0),
            "flood": (0, 0), "india": (20.593, 78.962),
        }
        text_lower = text.lower()
        for city, coords in CITY_COORDS.items():
            if city in text_lower and coords != (0, 0):
                return (
                    round(coords[0] + random.uniform(-0.5, 0.5), 4),
                    round(coords[1] + random.uniform(-0.5, 0.5), 4),
                )
        return None, None

    def start_background(self) -> threading.Thread:
        self.running = True
        t = threading.Thread(target=self.start, daemon=True)
        t.start()
        return t

    def stop(self):
        self.running = False


# ─────────────────────────────────────────────────────────────────────────────
# 3. TWITTER/X PRODUCER  (REFERENCE — $5000/month Pro tier required)
# ─────────────────────────────────────────────────────────────────────────────

class TwitterProducer:
    """
    Twitter/X Filtered Stream API v2.
    REQUIRES: Pro tier subscription ($5000/month) for streaming.

    Free tier: ~1,500 posts/month read-only — NO streaming.
    Basic tier ($200/month): search only, NO streaming.
    Pro tier ($5000/month): Filtered Stream access.

    For MTech research, use Bluesky or Mastodon instead.
    This class is kept for reference / future use.

    API docs: https://developer.x.com/en/docs/x-api/tweets/filtered-stream
    """

    RULES = [
        ("(flood OR flooding) has:geo lang:en",       "flood"),
        ("(earthquake OR quake) has:geo lang:en",     "earthquake"),
        ("(wildfire OR bushfire) has:geo lang:en",    "wildfire"),
        ("(cyclone OR hurricane) has:geo lang:en",    "cyclone"),
        ("(landslide OR mudslide) has:geo lang:en",   "landslide"),
    ]

    def __init__(self, kafka_producer: KafkaProducerWrapper):
        self.kafka        = kafka_producer
        self.bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
        self.running      = False

    def start(self):
        if not self.bearer_token:
            logger.error(
                "TWITTER_BEARER_TOKEN not set. "
                "Note: Streaming requires Pro tier ($5000/month). "
                "Use --source bluesky instead."
            )
            return

        try:
            import tweepy
        except ImportError:
            logger.error("tweepy not installed: pip install tweepy")
            return

        class _StreamClient(tweepy.StreamingClient):
            def __init__(self_, bearer, kafka, topic):
                super().__init__(bearer)
                self_._kafka = kafka
                self_._topic = topic

            def on_tweet(self_, tweet):
                text = tweet.text
                if not is_disaster_relevant(text):
                    return
                data = {
                    "text":      text,
                    "lat":       None,
                    "lon":       None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source":    "twitter",
                }
                if tweet.geo and tweet.geo.get("coordinates"):
                    c = tweet.geo["coordinates"]["coordinates"]
                    data["lon"], data["lat"] = c[0], c[1]
                if data["lat"]:
                    self_._kafka.send(self_._topic, data)

        client = _StreamClient(self.bearer_token, self.kafka, KAFKA_TOPIC)

        # Clear and reset filter rules
        existing = client.get_rules()
        if existing.data:
            client.delete_rules([r.id for r in existing.data])
        client.add_rules([tweepy.StreamRule(v, tag=t) for v, t in self.RULES])

        self.running = True
        logger.info("Twitter stream started (Pro tier required)")
        client.filter(
            tweet_fields=["created_at", "geo", "public_metrics"],
            expansions=["geo.place_id"],
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. MOCK PRODUCER  (No dependencies — always works)
# ─────────────────────────────────────────────────────────────────────────────

class MockSocialProducer:
    """
    Generates realistic synthetic disaster posts at regular intervals.
    Use this when you have no API access or are testing offline.
    The synthetic data has the same schema as real posts.
    """

    TEMPLATES = [
        ("Massive flooding reported in {city}! Roads completely submerged. #flood #disaster",    19.076, 72.877, "flood"),
        ("Flash floods hit {city}. Emergency services deployed. #FloodAlert",                     28.613, 77.209, "flood"),
        ("BREAKING: Earthquake shakes {city}! Magnitude 6.2 reported. #earthquake",              37.900, 32.860, "earthquake"),
        ("Buildings shaking in {city} after strong tremor. #quake #earthquake",                   13.082, 80.270, "earthquake"),
        ("Wildfire spreading near {city}. Residents evacuating. #wildfire",                       34.052,-118.243,"wildfire"),
        ("Massive blaze engulfs forest near {city}. Air quality critical. #WildfireSeason",       37.774,-122.419,"wildfire"),
        ("Cyclone warning issued for {city} coast. Category 4. #CycloneAlert",                    20.300, 85.820, "cyclone"),
        ("Storm surge expected as cyclone approaches {city}. Evacuate immediately! #cyclone",     13.090, 80.270, "cyclone"),
        ("Landslide blocks national highway near {city}. People trapped. #landslide",             27.700, 85.318, "landslide"),
        ("Traffic disrupted in {city} after landslide. Rescue operations underway. #mudslide",    12.971, 77.594, "landslide"),
        ("Normal day in {city}. Nothing to report.",                                               20.000, 77.000, "none"),
        ("Great weather today in {city}! Going for a walk.",                                       19.076, 72.877, "none"),
    ]

    CITIES = ["Mumbai", "Delhi", "Chennai", "Kolkata", "Hyderabad",
              "Bangalore", "Lahore", "Dhaka", "Kathmandu", "Istanbul",
              "Tokyo", "Jakarta", "Manila", "Los Angeles", "Sydney"]

    def __init__(self, kafka_producer: KafkaProducerWrapper,
                 interval: float = 2.0, disaster_ratio: float = 0.7):
        self.kafka          = kafka_producer
        self.interval       = interval
        self.disaster_ratio = disaster_ratio
        self.running        = False
        self._count         = 0

    def start(self):
        self.running = True
        logger.info(f"Mock social producer started (interval={self.interval}s)")

        while self.running:
            city    = random.choice(self.CITIES)
            # Weight towards disaster posts
            pool    = [t for t in self.TEMPLATES if t[3] != "none"] \
                      if random.random() < self.disaster_ratio \
                      else [t for t in self.TEMPLATES if t[3] == "none"]

            tmpl, base_lat, base_lon, label = random.choice(pool)
            text = tmpl.format(city=city)

            if not is_disaster_relevant(text) and label != "none":
                continue

            lat = round(base_lat + random.uniform(-1.5, 1.5), 4)
            lon = round(base_lon + random.uniform(-1.5, 1.5), 4)

            message = {
                "text":             text,
                "lat":              lat,
                "lon":              lon,
                "timestamp":        datetime.now(timezone.utc).isoformat(),
                "author_id":        str(random.randint(10000, 9999999)),
                "followers_count":  random.randint(50, 80000),
                "retweet_count":    random.randint(0, 500) if label != "none" else 0,
                "verified":         random.random() < 0.05,
                "account_age_days": random.randint(30, 2000),
                "source":           "mock",
                "platform":         "mock",
                "ground_truth_label": label,   # useful for evaluation
            }

            self.kafka.send(KAFKA_TOPIC, message, key=f"{lat:.2f}_{lon:.2f}")
            self._count += 1

            if self._count % 20 == 0:
                logger.info(f"Mock producer: sent {self._count} posts")

            time.sleep(self.interval)

    def start_background(self) -> threading.Thread:
        self.running = True
        t = threading.Thread(target=self.start, daemon=True)
        t.start()
        return t

    def stop(self):
        self.running = False


# ─────────────────────────────────────────────────────────────────────────────
# Factory function
# ─────────────────────────────────────────────────────────────────────────────

def create_producer(source: str, kafka_producer: KafkaProducerWrapper):
    """Return the appropriate producer based on source name."""
    sources = {
        "bluesky":  BlueskyJetstreamProducer,
        "mastodon": MastodonStreamProducer,
        "twitter":  TwitterProducer,
        "mock":     MockSocialProducer,
    }
    if source not in sources:
        raise ValueError(f"Unknown source '{source}'. Choose: {list(sources.keys())}")
    return sources[source](kafka_producer)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Social media stream → Kafka")
    parser.add_argument(
        "--source",
        choices=["bluesky", "mastodon", "twitter", "mock"],
        default="bluesky",
        help="Data source. Use 'bluesky' (free) or 'mock' (offline testing)."
    )
    parser.add_argument("--interval", type=float, default=2.0,
                        help="Mock producer interval in seconds")
    args = parser.parse_args()

    kafka = KafkaProducerWrapper()
    prod  = create_producer(args.source, kafka)

    if args.source == "mock":
        prod.interval = args.interval

    logger.info(f"Starting {args.source} producer → Kafka topic '{KAFKA_TOPIC}'")
    logger.info("Press Ctrl+C to stop.")

    try:
        prod.start()
    except KeyboardInterrupt:
        logger.info("Stopping producer...")
        prod.stop()
        kafka.flush()
        kafka.close()
        logger.success("Done.")
    