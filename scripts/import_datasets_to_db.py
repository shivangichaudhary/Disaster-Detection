"""
scripts/import_datasets_to_db.py
──────────────────────────────
Connects and inserts preprocessed dataset events (Sentinel-1, BigEarthNet-SAR,
CrisisMMD, HumAID, CREDBANK) into the PostgreSQL/PostGIS 'disaster_alerts' database.
"""

import os
import sys
import json
import yaml
from pathlib import Path
import pandas as pd
from loguru import logger
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    logger.info("=== Importing Ingested Datasets to PostGIS Database ===")

    # Load configuration
    config_path = Path("configs/train_config.yaml")
    if not config_path.exists():
        logger.error(f"Configuration file not found at {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # 1. Read Database configuration
    from dotenv import load_dotenv
    load_dotenv()
    db_url = os.getenv("DATABASE_URL", "postgresql://disaster:disaster123@localhost:5432/disaster_db")

    try:
        from sqlalchemy import create_engine, text
        from sqlalchemy.orm import sessionmaker
        engine = create_engine(db_url)
        # Verify connection immediately
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        Session = sessionmaker(bind=engine)
        logger.info(f"Connected to database: {engine.url.database} on {engine.url.host}")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        logger.warning("Make sure the Postgres container is running. Use 'docker-compose up -d'.")
        sys.exit(1)

    # 2. Read preprocessed files
    labels_file = Path(cfg["data"]["label_file"])
    tweets_file = Path(cfg["data"]["tweet_root"])

    if not labels_file.exists():
        logger.error(f"Labels file not found at {labels_file}. Run 'preprocess_all.py' first.")
        sys.exit(1)

    labels_df = pd.read_csv(labels_file)
    logger.info(f"Loaded {len(labels_df)} dataset records to import.")

    # Read tweets if available to count co-located events
    tweet_counts = {}
    if tweets_file.exists():
        tweets_df = pd.read_csv(tweets_file)
        if "geohash" in tweets_df.columns:
            # Count tweets per geohash (first 4 characters for regional clustering)
            tweets_df["geohash_prefix"] = tweets_df["geohash"].astype(str).str[:4]
            tweet_counts = tweets_df["geohash_prefix"].value_counts().to_dict()
            logger.info(f"Loaded {len(tweets_df)} tweets for spatial density estimation.")
    else:
        logger.warning(f"Tweets processed file not found at {tweets_file}. Importing without tweet counts.")

    # 3. Insert records
    import_count = 0
    with Session() as session:
        for _, row in labels_df.iterrows():
            try:
                lat = float(row["lat"])
                lon = float(row["lon"])
                disaster_type = str(row["label"])
                severity_id = int(row.get("label_severity", 0))
                severity = ["low", "medium", "high"][min(severity_id, 2)]
                
                # Fetch confidence
                confidence = 0.95 if disaster_type != "none" else 0.0
                is_disaster = bool(disaster_type != "none")

                # Generate timestamp
                ts_str = row.get("timestamp")
                if pd.isna(ts_str) or not ts_str:
                    ts = datetime.now(timezone.utc)
                else:
                    try:
                        ts = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
                    except Exception:
                        ts = datetime.now(timezone.utc)

                # Map geohash
                from utils.tweet_preprocessing import to_geohash
                gh = to_geohash(lat, lon, precision=5)
                gh_prefix = gh[:4] if gh else ""

                # Count co-located tweets
                t_count = tweet_counts.get(gh_prefix, 0)

                raw_json = {
                    "source": row.get("source", "unknown"),
                    "label_id": int(row.get("label_id", 5)),
                    "split": row.get("split", "train"),
                    "filepath": row.get("filepath", "")
                }

                session.execute(text("""
                    INSERT INTO disaster_alerts
                    (timestamp, lat, lon, geom, disaster_type, severity, confidence, is_disaster, tweet_count, geohash, raw_json)
                    VALUES (
                        :ts, :lat, :lon,
                        ST_SetSRID(ST_MakePoint(:lon, :lat), 4326),
                        :dtype, :severity, :conf, :is_disaster, :t_count, :geohash, :raw_json::jsonb
                    )
                """), {
                    "ts": ts,
                    "lat": lat,
                    "lon": lon,
                    "dtype": disaster_type,
                    "severity": severity,
                    "conf": confidence,
                    "is_disaster": is_disaster,
                    "t_count": t_count,
                    "geohash": gh,
                    "raw_json": json.dumps(raw_json)
                })
                import_count += 1
            except Exception as e:
                logger.warning(f"Failed to import row: {e}")

        session.commit()

    logger.success(f"Successfully connected and imported {import_count} dataset records to PostGIS database.")

if __name__ == "__main__":
    main()
