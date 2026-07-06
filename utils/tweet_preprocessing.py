"""
utils/tweet_preprocessing.py
─────────────────────────────
Twitter/X stream preprocessing pipeline:
  • Text cleaning (URLs, mentions, emojis)
  • Language detection and filtering
  • Geolocation extraction / geohash binning
  • Bot detection heuristics
  • Credibility scoring
  • Temporal windowing for TAM alignment
"""

import re
import math
import hashlib
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
from loguru import logger

try:
    import pygeohash as pgh
    GEOHASH_AVAILABLE = True
except ImportError:
    GEOHASH_AVAILABLE = False

try:
    from langdetect import detect as detect_lang
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

# ── Disaster keyword lexicon ──────────────────────────────────────────────────

DISASTER_KEYWORDS = {
    "flood":      [
        "flood", "flooding", "floods", "inundation", "waterlogged",
        "flash flood", "river overflow", "submerged", "drowning", "deluge",
    ],
    "earthquake": [
        "earthquake", "quake", "tremor", "aftershock", "seismic",
        "richter", "epicenter", "magnitude", "fault line", "shake",
    ],
    "wildfire":   [
        "wildfire", "forest fire", "bushfire", "blaze", "inferno",
        "evacuate", "smoke", "ash", "burning", "firestorm",
    ],
    "cyclone":    [
        "cyclone", "hurricane", "typhoon", "storm surge", "landfall",
        "tropical storm", "wind speed", "storm warning", "Category",
    ],
    "landslide":  [
        "landslide", "mudslide", "rockfall", "debris flow",
        "slope failure", "avalanche", "mudflow",
    ],
}

ALL_DISASTER_KEYWORDS = [kw for kws in DISASTER_KEYWORDS.values() for kw in kws]


# ── Text Cleaning ─────────────────────────────────────────────────────────────

class TweetCleaner:
    """Clean raw tweet text for NLP processing."""

    URL_RE       = re.compile(r"https?://\S+|www\.\S+")
    MENTION_RE   = re.compile(r"@\w+")
    HASHTAG_RE   = re.compile(r"#(\w+)")
    RT_RE        = re.compile(r"^RT\s+")
    EMOJI_RE     = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    MULTI_SPACE  = re.compile(r"\s+")
    PUNCT_RE     = re.compile(r"[^\w\s#@!?.,'\"]+")

    def clean(
        self,
        text: str,
        keep_hashtags: bool = True,
        keep_mentions: bool = False,
    ) -> str:
        text = self.RT_RE.sub("", text)
        text = self.URL_RE.sub("[URL]", text)
        if not keep_mentions:
            text = self.MENTION_RE.sub("", text)
        if keep_hashtags:
            # Keep hashtag content without the # sign
            text = self.HASHTAG_RE.sub(r"\1", text)
        else:
            text = self.HASHTAG_RE.sub("", text)
        text = self.EMOJI_RE.sub(" ", text)
        text = self.MULTI_SPACE.sub(" ", text).strip()
        return text

    def is_disaster_relevant(self, text: str, threshold: int = 1) -> bool:
        """Check if tweet contains disaster keywords."""
        text_lower = text.lower()
        count = sum(1 for kw in ALL_DISASTER_KEYWORDS if kw in text_lower)
        return count >= threshold

    def get_disaster_type(self, text: str) -> Optional[str]:
        """Return most likely disaster type from tweet text."""
        text_lower = text.lower()
        scores = {}
        for dtype, keywords in DISASTER_KEYWORDS.items():
            scores[dtype] = sum(1 for kw in keywords if kw in text_lower)
        if max(scores.values()) == 0:
            return None
        return max(scores, key=scores.get)


# ── Bot Detection ─────────────────────────────────────────────────────────────

class BotDetector:
    """
    Heuristic bot detection based on account and tweet features.
    Assigns a bot probability score [0, 1].
    """

    def score(self, tweet_meta: dict) -> float:
        """
        Args:
            tweet_meta: dict with keys:
                followers_count, friends_count, statuses_count,
                account_age_days, default_profile, verified,
                tweet_rate_per_day (computed)
        Returns:
            bot_probability in [0, 1]. > 0.7 = likely bot.
        """
        red_flags = 0
        total = 8

        followers = tweet_meta.get("followers_count", 0)
        friends   = tweet_meta.get("friends_count", 1)
        statuses  = tweet_meta.get("statuses_count", 0)
        age_days  = tweet_meta.get("account_age_days", 1)
        rate      = statuses / max(age_days, 1)

        # High follower/friend ratio (buying followers)
        if friends > 0 and followers / friends > 100:
            red_flags += 1

        # Very high tweet rate (> 100/day)
        if rate > 100:
            red_flags += 2

        # Very new account (< 30 days) with many tweets
        if age_days < 30 and statuses > 500:
            red_flags += 2

        # No followers but very active
        if followers < 5 and statuses > 100:
            red_flags += 1

        # Default profile (never updated)
        if tweet_meta.get("default_profile", False):
            red_flags += 1

        # Verified accounts are almost never bots
        if tweet_meta.get("verified", False):
            red_flags = max(0, red_flags - 2)

        return min(red_flags / total, 1.0)

    def is_bot(self, tweet_meta: dict, threshold: float = 0.7) -> bool:
        return self.score(tweet_meta) >= threshold


# ── Credibility Scoring ───────────────────────────────────────────────────────

class CredibilityScorer:
    """
    Multi-factor credibility scorer for disaster tweets.
    Based on: Castillo et al. (2011) "Information credibility on Twitter"
    and CREDBANK credibility labels.
    """

    def __init__(self):
        self.cleaner = TweetCleaner()
        self.bot_detector = BotDetector()

    def score(self, tweet: dict) -> float:
        """
        Compute credibility score [0, 1] for a tweet.
        Higher = more credible disaster signal.
        """
        scores = {}

        # 1. Content features (40% weight)
        text = tweet.get("text", "")
        cleaned = self.cleaner.clean(text)
        has_disaster_kw   = self.cleaner.is_disaster_relevant(text)
        has_numbers       = bool(re.search(r"\d+", text))
        has_location      = bool(tweet.get("lat") and tweet.get("lon"))
        has_url           = "[URL]" in cleaned or "http" in text
        has_media         = tweet.get("has_media", False)

        content_score = (
            0.40 * float(has_disaster_kw) +
            0.15 * float(has_numbers) +
            0.20 * float(has_location) +
            0.15 * float(has_media) +
            0.10 * float(has_url)
        )
        scores["content"] = content_score

        # 2. User features (30% weight)
        followers    = tweet.get("followers_count", 0)
        verified     = tweet.get("verified", False)
        age_days     = tweet.get("account_age_days", 30)
        bot_score    = self.bot_detector.score(tweet)

        follower_norm  = min(math.log1p(followers) / math.log1p(100000), 1.0)
        user_score = (
            0.40 * follower_norm +
            0.30 * float(verified) +
            0.20 * min(age_days / 365, 1.0) +
            0.10 * (1.0 - bot_score)
        )
        scores["user"] = user_score

        # 3. Propagation / retweet features (20% weight)
        rt_count   = tweet.get("retweet_count", 0)
        reply_count= tweet.get("reply_count", 0)
        like_count = tweet.get("like_count", 0)

        prop_score = min(
            0.5 * math.log1p(rt_count) / math.log1p(1000) +
            0.3 * math.log1p(like_count) / math.log1p(5000) +
            0.2 * math.log1p(reply_count) / math.log1p(500),
            1.0,
        )
        scores["propagation"] = prop_score

        # 4. Temporal relevance (10% weight): recent tweets score higher
        timestamp = tweet.get("timestamp")
        if timestamp:
            try:
                if isinstance(timestamp, str):
                    ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                else:
                    ts = timestamp
                age_hours = (datetime.now(ts.tzinfo) - ts).total_seconds() / 3600
                temporal_score = max(0, 1.0 - age_hours / 720)  # decay over 30 days
            except Exception:
                temporal_score = 0.5
        else:
            temporal_score = 0.5
        scores["temporal"] = temporal_score

        # Weighted combination
        total = (
            0.40 * scores["content"] +
            0.30 * scores["user"] +
            0.20 * scores["propagation"] +
            0.10 * scores["temporal"]
        )

        # Bot penalty
        if bot_score > 0.7:
            total *= 0.1

        return round(min(max(total, 0.0), 1.0), 4)


# ── Geolocation & Geohash ─────────────────────────────────────────────────────

def extract_geolocation(tweet: dict) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract lat/lon from tweet metadata.
    Priority: coordinates > place bbox centroid > user location (NLP).
    """
    # Direct coordinates
    coords = tweet.get("coordinates") or tweet.get("geo")
    if coords:
        if isinstance(coords, dict):
            c = coords.get("coordinates", coords.get("coordinates"))
            if c and len(c) == 2:
                return float(c[1]), float(c[0])  # (lat, lon)

    # Place bounding box centroid
    place = tweet.get("place")
    if place and isinstance(place, dict):
        bb = place.get("bounding_box", {}).get("coordinates")
        if bb:
            lons = [p[0] for p in bb[0]]
            lats = [p[1] for p in bb[0]]
            return float(np.mean(lats)), float(np.mean(lons))

    # Fallback: stored lat/lon fields
    lat = tweet.get("lat") or tweet.get("latitude")
    lon = tweet.get("lon") or tweet.get("longitude")
    if lat and lon:
        return float(lat), float(lon)

    return None, None


def to_geohash(lat: float, lon: float, precision: int = 5) -> Optional[str]:
    """Encode lat/lon to geohash string for spatial grouping."""
    if not GEOHASH_AVAILABLE:
        # Simple fallback: tile string
        tile_lat = int(lat * 10) / 10
        tile_lon = int(lon * 10) / 10
        return f"{tile_lat:.1f}_{tile_lon:.1f}"
    return pgh.encode(lat, lon, precision=precision)


def geohash_to_bbox(geohash: str) -> Tuple[float, float, float, float]:
    """Return (min_lat, max_lat, min_lon, max_lon) for a geohash cell."""
    if GEOHASH_AVAILABLE:
        lat, lon, lat_err, lon_err = pgh.decode_exactly(geohash)
        return lat - lat_err, lat + lat_err, lon - lon_err, lon + lon_err
    # Fallback from our simple format
    parts = geohash.split("_")
    lat, lon = float(parts[0]), float(parts[1])
    return lat - 0.05, lat + 0.05, lon - 0.05, lon + 0.05


# ── Temporal Window Builder ───────────────────────────────────────────────────

class TemporalWindowBuilder:
    """
    Groups tweets into time windows aligned to SAR acquisition timestamps.
    This is the core of the Temporal Alignment Module (TAM) data preparation.
    """

    def __init__(self, window_minutes: int = 30, step_minutes: int = 5):
        self.window_minutes = window_minutes
        self.step_minutes   = step_minutes

    def build_windows(
        self,
        tweets: pd.DataFrame,
        sar_timestamps: List[datetime],
        geohash_precision: int = 5,
    ) -> Dict[str, List[pd.DataFrame]]:
        """
        For each SAR timestamp, gather tweets within the window
        in the same geohash region.

        Returns:
            Dict mapping sar_timestamp_str -> list of tweet DataFrames (per geohash)
        """
        windows = {}
        tweets["timestamp"] = pd.to_datetime(tweets["timestamp"], utc=True)

        for sar_ts in sar_timestamps:
            if sar_ts.tzinfo is None:
                sar_ts = sar_ts.replace(tzinfo=__import__("datetime").timezone.utc)

            window_start = sar_ts - timedelta(minutes=self.window_minutes)
            window_end   = sar_ts + timedelta(minutes=self.window_minutes)

            mask    = (tweets["timestamp"] >= window_start) & (tweets["timestamp"] <= window_end)
            in_win  = tweets[mask].copy()

            if in_win.empty:
                continue

            # Group by geohash
            geohash_groups = defaultdict(list)
            for _, row in in_win.iterrows():
                gh = row.get("geohash", "unknown")
                geohash_groups[gh].append(row)

            key = sar_ts.isoformat()
            windows[key] = {
                gh: pd.DataFrame(rows)
                for gh, rows in geohash_groups.items()
            }

        return windows

    def aggregate_window(self, tweet_window: pd.DataFrame) -> dict:
        """
        Aggregate tweet statistics over a time window.
        Returns features for TAM input.
        """
        if tweet_window.empty:
            return {
                "tweet_count": 0,
                "mean_credibility": 0.0,
                "max_credibility": 0.0,
                "disaster_keyword_ratio": 0.0,
                "bot_ratio": 0.0,
            }

        return {
            "tweet_count":             len(tweet_window),
            "mean_credibility":        float(tweet_window.get("credibility_score", pd.Series([0])).mean()),
            "max_credibility":         float(tweet_window.get("credibility_score", pd.Series([0])).max()),
            "disaster_keyword_ratio":  float(
                tweet_window["text"].apply(
                    lambda t: TweetCleaner().is_disaster_relevant(t)
                ).mean()
            ),
            "unique_users":            int(tweet_window.get("user_id", pd.Series([])).nunique()),
        }


# ── Full Preprocessing Pipeline ───────────────────────────────────────────────

class TweetPreprocessor:
    """End-to-end tweet preprocessing pipeline."""

    def __init__(self, min_credibility: float = 0.3, lang: str = "en"):
        self.cleaner    = TweetCleaner()
        self.scorer     = CredibilityScorer()
        self.bot_det    = BotDetector()
        self.min_cred   = min_credibility
        self.lang       = lang

    def process_batch(self, tweets: List[dict]) -> pd.DataFrame:
        """
        Process a batch of raw tweet dicts.
        Returns cleaned, scored, filtered DataFrame.
        """
        records = []
        for tweet in tweets:
            try:
                text = tweet.get("text", "")
                if not text:
                    continue

                # Language filter
                if LANGDETECT_AVAILABLE:
                    try:
                        if detect_lang(text) != self.lang:
                            continue
                    except Exception:
                        pass

                # Clean text
                cleaned = self.cleaner.clean(text)

                # Geolocation
                lat, lon = extract_geolocation(tweet)
                if lat is None:
                    continue  # skip unlocated tweets

                geohash = to_geohash(lat, lon)

                # Bot check
                if self.bot_det.is_bot(tweet):
                    continue

                # Credibility score
                tweet["lat"] = lat
                tweet["lon"] = lon
                cred = self.scorer.score(tweet)

                if cred < self.min_cred:
                    continue

                records.append({
                    "text":             cleaned,
                    "raw_text":         text,
                    "lat":              lat,
                    "lon":              lon,
                    "geohash":          geohash,
                    "credibility_score": cred,
                    "disaster_type":    self.cleaner.get_disaster_type(text),
                    "timestamp":        tweet.get("created_at") or tweet.get("timestamp"),
                    "user_id":          tweet.get("author_id") or tweet.get("user_id"),
                    "followers_count":  tweet.get("followers_count", 0),
                    "retweet_count":    tweet.get("retweet_count", 0),
                })
            except Exception as e:
                logger.debug(f"Failed to process tweet: {e}")
                continue

        return pd.DataFrame(records)


if __name__ == "__main__":
    # Quick smoke test
    sample_tweets = [
        {
            "text": "Massive flooding in Mumbai! Roads completely submerged. #flood #Mumbai",
            "lat": 19.0760, "lon": 72.8777,
            "followers_count": 5420, "retweet_count": 234,
            "verified": False, "account_age_days": 720,
            "timestamp": "2023-09-15T14:30:00",
        },
        {
            "text": "Just had lunch in Delhi. Nice weather today!",
            "lat": 28.6139, "lon": 77.2090,
            "followers_count": 120, "retweet_count": 0,
            "verified": False, "account_age_days": 30,
            "timestamp": "2023-09-15T14:25:00",
        },
    ]

    proc = TweetPreprocessor(min_credibility=0.1)
    df   = proc.process_batch(sample_tweets)
    print(df[["text", "credibility_score", "disaster_type", "geohash"]].to_string())
