"""
scripts/download_datasets.py
────────────────────────────
Downloads all required datasets:
  1. BigEarthNet-SAR (SAR patches with labels)
  2. CrisisMMD      (multimodal crisis tweets)
  3. HumAID         (humanitarian tweet labels)
  4. CREDBANK       (tweet credibility labels)
  5. Sample Sentinel-1 tiles via sentinelsat
"""

import os
import json
import zipfile
import tarfile
import requests
import argparse
from pathlib import Path
from tqdm import tqdm
from loguru import logger


# ── Dataset registry ─────────────────────────────────────────────────────────
DATASETS = {
    "bigearth_sar": {
        "url": "https://bigearth.net/downloads/BigEarthNet-S1-v1.0.tar.gz",
        "dest": "data/raw/bigearth_sar",
        "desc": "BigEarthNet SAR patches (590k patches, multi-label)",
        "size_gb": 11.3,
    },
    "crisisMMD": {
        "url": "https://crisisnlp.qcri.org/data/crisismmd/CrisisMMD_v2.0.tar.gz",
        "dest": "data/raw/crisisMMD",
        "desc": "CrisisMMD - multimodal crisis tweets",
        "size_gb": 0.9,
    },
    "humaid": {
        "url": "https://crisisnlp.qcri.org/data/humaid/HumAID_data_v1.0.tar.gz",
        "dest": "data/raw/humaid",
        "desc": "HumAID - 77k labelled humanitarian tweets",
        "size_gb": 0.05,
    },
    "credbank": {
        "url": "https://figshare.com/ndownloader/files/6196852",
        "dest": "data/raw/credbank",
        "desc": "CREDBANK - tweet credibility annotations",
        "size_gb": 2.1,
    },
}

DISASTER_LABELS = {
    # BigEarthNet label index -> our label
    "Permanent water bodies": "flood",
    "Pastures": None,
    "Urban fabric": None,
    "Industrial or commercial units": None,
    "Road and rail networks": None,
    # Extended labels for disaster mapping
    "flood":       0,
    "earthquake":  1,
    "wildfire":    2,
    "cyclone":     3,
    "landslide":   4,
    "none":        5,
}


def download_file(url: str, dest_path: Path, desc: str = "") -> bool:
    """Stream-download a file with progress bar."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest_path, "wb") as f, tqdm(
            total=total, unit="iB", unit_scale=True, desc=desc
        ) as bar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        return True
    except Exception as e:
        logger.error(f"Download failed for {url}: {e}")
        return False


def extract_archive(archive_path: Path, dest_dir: Path):
    """Extract .tar.gz or .zip archives."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Extracting {archive_path.name} → {dest_dir}")
    if archive_path.suffix == ".gz" or str(archive_path).endswith(".tar.gz"):
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(dest_dir)
    elif archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(dest_dir)


def download_sentinel1_sample(output_dir: Path):
    """
    Download sample Sentinel-1 GRD products using sentinelsat.
    Requires ESA Copernicus Hub credentials in .env file:
      COPERNICUS_USER=your_username
      COPERNICUS_PASS=your_password

    Alternatively, use the open-access Copernicus Data Space Ecosystem.
    We provide 5 disaster event tiles (flood/earthquake/wildfire/cyclone).
    """
    from dotenv import load_dotenv
    load_dotenv()

    user = os.getenv("COPERNICUS_USER")
    pwd  = os.getenv("COPERNICUS_PASS")

    if not user or not pwd:
        logger.warning(
            "Copernicus credentials not found in .env. "
            "Skipping Sentinel-1 download. "
            "Set COPERNICUS_USER and COPERNICUS_PASS in .env file."
        )
        _create_synthetic_sar_samples(output_dir)
        return

    try:
        from sentinelsat import SentinelAPI
        from datetime import date

        api = SentinelAPI(user, pwd, "https://apihub.copernicus.eu/apihub")

        # Sample disaster event areas (bounding boxes)
        disaster_areas = [
            # Pakistan floods 2022
            {
                "footprint": "POLYGON((67.0 27.0, 70.0 27.0, 70.0 30.0, 67.0 30.0, 67.0 27.0))",
                "date": ("20220801", "20220930"),
                "label": "flood",
            },
            # Turkey earthquake 2023
            {
                "footprint": "POLYGON((36.0 37.0, 38.0 37.0, 38.0 38.5, 36.0 38.5, 36.0 37.0))",
                "date": ("20230201", "20230215"),
                "label": "earthquake",
            },
        ]

        output_dir.mkdir(parents=True, exist_ok=True)
        for area in disaster_areas:
            products = api.query(
                area["footprint"],
                date=area["date"],
                platformname="Sentinel-1",
                producttype="GRD",
            )
            if products:
                pid = list(products.keys())[0]
                logger.info(f"Downloading {area['label']} tile: {products[pid]['title']}")
                api.download(pid, directory_path=output_dir / area["label"])

    except ImportError:
        logger.warning("sentinelsat not installed. Using synthetic SAR data.")
        _create_synthetic_sar_samples(output_dir)


def _create_synthetic_sar_samples(output_dir: Path):
    """
    Create realistic synthetic SAR-like patches for testing without
    actual satellite data. Uses statistical properties of real SAR.
    """
    import numpy as np
    try:
        import rasterio
        from rasterio.transform import from_bounds
    except ImportError:
        logger.warning("rasterio not available. Skipping synthetic SAR generation.")
        return

    logger.info("Creating synthetic SAR patches for testing...")
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = ["flood", "earthquake", "wildfire", "cyclone", "none"]
    n_per_class = 50

    # SAR backscatter statistics per disaster type (dB scale)
    stats = {
        "flood":      {"mean": -15.0, "std": 3.0},  # low backscatter (water)
        "earthquake": {"mean": -8.0,  "std": 5.0},  # rubble = mixed
        "wildfire":   {"mean": -10.0, "std": 4.0},
        "cyclone":    {"mean": -12.0, "std": 6.0},
        "none":       {"mean": -6.0,  "std": 3.0},  # normal urban/veg
    }

    transform = from_bounds(0, 0, 1, 1, 256, 256)
    metadata = []

    for label in labels:
        label_dir = output_dir / label
        label_dir.mkdir(exist_ok=True)
        s = stats[label]
        for i in range(n_per_class):
            # Simulate 2-band SAR (VV, VH) with speckle noise
            vv = np.random.normal(s["mean"], s["std"], (256, 256)).astype(np.float32)
            vh = vv + np.random.normal(-3.0, 1.5, (256, 256)).astype(np.float32)
            # Add speckle
            speckle = np.random.gamma(1, 1, (256, 256)).astype(np.float32)
            vv = vv * speckle
            vh = vh * speckle

            fp = label_dir / f"patch_{i:04d}.tif"
            with rasterio.open(
                fp, "w", driver="GTiff",
                height=256, width=256, count=2,
                dtype="float32", crs="EPSG:4326",
                transform=transform,
            ) as dst:
                dst.write(vv, 1)
                dst.write(vh, 2)

            metadata.append({
                "filepath": str(fp),
                "label": label,
                "label_id": labels.index(label),
                "lat": float(np.random.uniform(20, 40)),
                "lon": float(np.random.uniform(60, 80)),
                "timestamp": f"2023-0{(i%9)+1}-{(i%28)+1:02d}T00:00:00",
            })

    import csv
    meta_file = output_dir / "sar_metadata.csv"
    with open(meta_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metadata[0].keys())
        writer.writeheader()
        writer.writerows(metadata)

    logger.info(f"Created {len(metadata)} synthetic SAR patches → {output_dir}")


def create_synthetic_tweets(output_dir: Path):
    """Create synthetic labelled disaster tweets for testing."""
    import pandas as pd
    import random

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating synthetic disaster tweets...")

    templates = {
        "flood": [
            "Roads completely flooded in {city}. People stuck on rooftops. #flood #disaster",
            "Water levels rising fast near {city}. Evacuation underway. #flooding",
            "Severe flooding reported in {city} district. Please avoid low-lying areas. #flood",
            "BREAKING: Flash floods hit {city}. Emergency services deployed. #FloodAlert",
            "Flood waters have submerged entire neighborhoods in {city}. #SOS #flood",
        ],
        "earthquake": [
            "Strong earthquake felt in {city}! Buildings shaking. #earthquake #tremor",
            "Magnitude 6.5 earthquake hits {city}. Casualties feared. #quake",
            "Aftershocks continue in {city} following major earthquake. #earthquake",
            "Rescue teams searching for survivors after earthquake in {city}. #disasterrelief",
            "Several buildings collapsed in {city} after earthquake. #earthquake #emergency",
        ],
        "wildfire": [
            "Massive wildfire burning near {city}. Residents evacuating. #wildfire #fire",
            "Fire spreading rapidly towards {city} residential areas. #WildfireAlert",
            "Smoke visible from {city} as wildfire grows. Firefighters on scene. #wildfire",
            "Thousands flee as wildfire approaches {city}. #WildFireEvacuation",
            "Wildfire destroying homes near {city}. Air quality critical. #wildfire",
        ],
        "cyclone": [
            "Cyclone approaching {city}! Storm surge expected. #cyclone #storm",
            "Category 4 hurricane/cyclone warning for {city} region. #CycloneAlert",
            "Extreme winds and rain battering {city} as cyclone makes landfall. #cyclone",
            "Cyclone damage reports from {city}. Power lines down. #disaster",
            "Cyclone intensity increasing. {city} under red alert. #CycloneWarning",
        ],
        "none": [
            "Beautiful day in {city} today! #weather",
            "Traffic jam on highway near {city}. #traffic",
            "Festival celebrations in {city} this weekend. #festival",
            "New restaurant opening in {city} downtown. #food",
            "Sports match results from {city}. #sports",
        ],
    }

    cities = [
        "Mumbai", "Chennai", "Kolkata", "Delhi", "Hyderabad",
        "Lahore", "Dhaka", "Kathmandu", "Colombo", "Karachi",
        "Bangkok", "Jakarta", "Manila", "Tokyo", "Istanbul",
    ]

    records = []
    for label, tmplates in templates.items():
        for i in range(200):
            city   = random.choice(cities)
            text   = random.choice(tmplates).format(city=city)
            lat    = round(random.uniform(8.0, 37.0), 4)
            lon    = round(random.uniform(68.0, 140.0), 4)
            cred   = round(random.uniform(0.4, 1.0) if label != "none" else random.uniform(0.1, 0.9), 3)
            records.append({
                "text": text,
                "label": label,
                "label_id": list(templates.keys()).index(label),
                "lat": lat,
                "lon": lon,
                "timestamp": f"2023-{random.randint(1,12):02d}-{random.randint(1,28):02d}T{random.randint(0,23):02d}:00:00",
                "credibility_score": cred,
                "user_followers": random.randint(100, 100000),
                "is_bot": False,
            })

    df = pd.DataFrame(records)
    df.to_csv(output_dir / "tweets_labelled.csv", index=False)
    logger.info(f"Generated {len(records)} synthetic tweets → {output_dir}")
    return df


def main(args):
    base = Path("data/raw")
    base.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("all", "sar"):
        logger.info("=== Setting up SAR data ===")
        download_sentinel1_sample(Path("data/raw/sentinel1"))

    if args.dataset in ("all", "bigearth"):
        logger.info("=== BigEarthNet-SAR ===")
        ds = DATASETS["bigearth_sar"]
        dest = Path(ds["dest"])
        archive = dest.parent / "bigearth_sar.tar.gz"
        logger.info(f"Downloading {ds['desc']} (~{ds['size_gb']} GB)")
        if download_file(ds["url"], archive, ds["desc"]):
            extract_archive(archive, dest)

    if args.dataset in ("all", "tweets"):
        logger.info("=== Setting up Tweet data ===")
        for name in ["crisisMMD", "humaid", "credbank"]:
            ds = DATASETS[name]
            dest = Path(ds["dest"])
            archive = dest.parent / f"{name}.tar.gz"
            logger.info(f"Downloading {ds['desc']} (~{ds['size_gb']} GB)")
            if not download_file(ds["url"], archive, ds["desc"]):
                logger.warning(f"Failed to download {name}. Creating synthetic data.")

        # Always create synthetic data as fallback / augmentation
        create_synthetic_tweets(Path("data/raw/synthetic_tweets"))

    if args.dataset in ("all", "synthetic"):
        logger.info("=== Creating synthetic datasets for testing ===")
        _create_synthetic_sar_samples(Path("data/raw/synthetic_sar"))
        create_synthetic_tweets(Path("data/raw/synthetic_tweets"))
        logger.success("Synthetic datasets created successfully!")

    logger.success("Dataset setup complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download disaster detection datasets")
    parser.add_argument(
        "--dataset",
        choices=["all", "sar", "bigearth", "tweets", "synthetic"],
        default="synthetic",
        help="Which dataset(s) to download. Use 'synthetic' for quick testing.",
    )
    args = parser.parse_args()
    main(args)
