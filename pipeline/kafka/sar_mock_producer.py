"""
pipeline/kafka/sar_mock_producer.py
────────────────────────────────────
Mock SAR (Sentinel-1) data producer for fusion-mode demo and offline testing.

Generates realistic synthetic SAR acquisition metadata and publishes it to
the 'sar-raw' Kafka topic at configurable intervals, simulating the cadence
at which new Sentinel-1 scenes would arrive in a real deployment.

Synthetic SAR patches are written as temporary .npy files so the SAR
preprocessing UDFs in the Spark pipeline (sar_stats_udf, sar_quality_udf)
have actual files to read and process.

Usage:
    # Stand-alone
    python pipeline/kafka/sar_mock_producer.py

    # From the launcher
    python scripts/run_spark_streaming.py --mode fusion --source mock
"""

from __future__ import annotations

import os
import sys
import json
import time
import random
import threading
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

# Project root
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.kafka.producers import KafkaProducerWrapper

KAFKA_TOPIC = "sar-raw"

# ── Synthetic SAR scene catalogue ─────────────────────────────────────────────
# (label, lat_centre, lon_centre, vv_mean_db, vh_mean_db, description)
SAR_CATALOGUE = [
    ("flood",      19.076,  72.877, -18.0, -21.0, "Mumbai coastal flooding"),
    ("flood",      28.613,  77.209, -17.5, -20.5, "Delhi river basin inundation"),
    ("earthquake", 37.900,  32.860,  -8.0, -11.0, "Turkey central fault zone"),
    ("earthquake", 13.082,  80.270,  -9.0, -12.0, "Chennai seismic event"),
    ("wildfire",   34.052,-118.243,  -6.0,  -9.0, "Los Angeles wildfire scar"),
    ("wildfire",   37.774,-122.419,  -5.5,  -8.5, "San Francisco area wildfire"),
    ("cyclone",    20.300,  85.820, -14.0, -17.0, "Odisha cyclone landfall"),
    ("cyclone",    13.090,  80.270, -15.0, -18.0, "Tamil Nadu storm surge"),
    ("landslide",  27.700,  85.318, -10.0, -13.0, "Nepal Himalayas slope failure"),
    ("landslide",  12.971,  77.594,  -9.5, -12.5, "Bangalore hills rockfall"),
    ("none",       20.000,  78.000,  -6.0,  -9.0, "Central India baseline"),
    ("none",       15.000,  74.000,  -7.0, -10.0, "Goa baseline clear"),
]

LABEL_NAMES = ["flood", "earthquake", "wildfire", "cyclone", "landslide", "none"]


class SARMockProducer:
    """
    Generates synthetic SAR acquisition events and publishes them to Kafka.

    Each event:
      1. Picks a random scene from the catalogue.
      2. Writes a synthetic NumPy patch (.npy) to a temp directory.
      3. Publishes metadata JSON to the 'sar-raw' Kafka topic.

    The Spark pipeline's sar_stats_udf reads the .npy file via
    utils.sar_preprocessing.SARPatchReader (Lee filter → dB → normalise).
    """

    def __init__(
        self,
        kafka_producer: KafkaProducerWrapper,
        interval: float = 15.0,           # SAR scenes arrive every 15 s in demo
        disaster_ratio: float = 0.70,
        output_dir: Optional[str] = None,
    ):
        self.kafka         = kafka_producer
        self.interval      = interval
        self.disaster_ratio = disaster_ratio
        self.output_dir    = Path(output_dir or tempfile.mkdtemp(prefix="sar_mock_"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.running       = False
        self._count        = 0

        logger.info(f"[SARMockProducer] Synthetic SAR files → {self.output_dir}")

    # ── Synthetic patch generation ────────────────────────────────────────────

    def _make_patch(self, label: str, vv_mean: float, vh_mean: float) -> Path:
        """
        Generate a synthetic 2-band SAR patch (VV, VH) as a .npy file.
        Statistics match the label's typical backscatter profile.
        """
        import numpy as np

        # Realistic std deviations for each class
        std_map = {
            "flood": 2.5, "earthquake": 4.5, "wildfire": 3.5,
            "cyclone": 5.0, "landslide": 4.0, "none": 3.0,
        }
        std = std_map.get(label, 3.5)

        size = 256
        vv = np.random.normal(vv_mean, std, (size, size)).astype("float32")
        vh = np.random.normal(vh_mean, std, (size, size)).astype("float32")
        patch = np.stack([vv, vh])           # shape: (2, 256, 256)

        fname = f"{label}_{self._count:06d}_{int(time.time())}.npy"
        fpath = self.output_dir / label
        fpath.mkdir(exist_ok=True)
        full_path = fpath / fname
        np.save(str(full_path), patch)
        return full_path

    # ── Main loop ─────────────────────────────────────────────────────────────

    def start(self):
        """Start producing SAR events (blocking)."""
        self.running = True
        logger.info(
            f"[SARMockProducer] Started  interval={self.interval}s  "
            f"disaster_ratio={self.disaster_ratio}"
        )

        while self.running:
            # Choose a scene
            if random.random() < self.disaster_ratio:
                pool = [s for s in SAR_CATALOGUE if s[0] != "none"]
            else:
                pool = [s for s in SAR_CATALOGUE if s[0] == "none"]

            label, lat, lon, vv_mean, vh_mean, desc = random.choice(pool)

            # Add spatial jitter
            lat += random.uniform(-0.8, 0.8)
            lon += random.uniform(-0.8, 0.8)

            # Write synthetic patch
            try:
                fpath = self._make_patch(label, vv_mean, vh_mean)
                filepath_str = str(fpath)
            except Exception as exc:
                logger.warning(f"[SARMockProducer] Failed to write patch: {exc}")
                filepath_str = f"synthetic://{label}/scene_{self._count:06d}.tif"

            # Build Kafka message matching SAR_SCHEMA
            message = {
                "filepath":   filepath_str,
                "filename":   Path(filepath_str).name,
                "label":      label,
                "timestamp":  datetime.now(timezone.utc).isoformat(),
                "size_bytes": 256 * 256 * 2 * 4,   # (H × W × bands × float32)
                "source":     "mock_sar",
                "lat":        round(lat, 4),
                "lon":        round(lon, 4),
                "description": desc,
                "vv_mean_approx": round(vv_mean + random.uniform(-1, 1), 2),
                "vh_mean_approx": round(vh_mean + random.uniform(-1, 1), 2),
            }

            key = f"{round(lat, 1)}_{round(lon, 1)}"
            self.kafka.send(KAFKA_TOPIC, message, key=key)
            self._count += 1

            if self._count % 10 == 0:
                logger.info(
                    f"[SARMockProducer] Sent {self._count} SAR events  "
                    f"last={label}@({lat:.2f},{lon:.2f})"
                )

            time.sleep(self.interval)

    def start_background(self) -> threading.Thread:
        """Start in a daemon background thread."""
        self.running = True
        t = threading.Thread(target=self.start, daemon=True, name="sar-mock-producer")
        t.start()
        return t

    def stop(self):
        self.running = False

    def cleanup(self):
        """Remove temporary .npy files created during this run."""
        import shutil
        try:
            shutil.rmtree(str(self.output_dir))
            logger.info(f"[SARMockProducer] Cleaned up {self.output_dir}")
        except Exception as exc:
            logger.warning(f"[SARMockProducer] Cleanup failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mock SAR producer → Kafka sar-raw")
    parser.add_argument("--interval",       type=float, default=15.0,
                        help="Seconds between SAR events (default: 15)")
    parser.add_argument("--disaster-ratio", type=float, default=0.70,
                        help="Fraction of events that are disasters (default: 0.70)")
    parser.add_argument("--output-dir",     default=None,
                        help="Directory to write synthetic .npy files")
    args = parser.parse_args()

    kafka = KafkaProducerWrapper()
    prod  = SARMockProducer(
        kafka,
        interval=args.interval,
        disaster_ratio=args.disaster_ratio,
        output_dir=args.output_dir,
    )

    logger.info(f"Starting mock SAR producer → topic '{KAFKA_TOPIC}'. Ctrl-C to stop.")
    try:
        prod.start()
    except KeyboardInterrupt:
        logger.info("Stopping SAR producer...")
        prod.stop()
        kafka.flush()
        kafka.close()
        prod.cleanup()
        logger.success("Done.")
