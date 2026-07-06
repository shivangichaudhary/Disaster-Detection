# Real-Time Disaster Detection via SAR + Social Media Fusion
**MTech CSE Final Year Project**

## Project Overview
A real-time big-data system integrating Sentinel-1 SAR satellite imagery and Twitter/X streams
through deep cross-modal learning for disaster detection and emergency response.

## Architecture
```
Sentinel-1 SAR ──► SAR Encoder (ViT/ResNet) ──►
                                                  TAM ──► Cross-Attention ──► Classifier ──► Alert
Twitter Stream ──► Text Encoder (BERTweet) ──►
```

## Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download datasets
python scripts/download_datasets.py

# 3. Preprocess data
python scripts/preprocess_all.py

# 4. Train model
python training/train.py --config configs/train_config.yaml

# 5. Run inference server
python inference/serve.py

# 6. Start Kafka pipeline (requires Docker)
docker-compose up -d
python pipeline/kafka/producer_sar.py &
python pipeline/kafka/producer_twitter.py &
python pipeline/spark/streaming_job.py
```

## Dataset Sources
| Dataset | Modality | Link |
|---------|----------|------|
| Sentinel-1 | SAR | https://scihub.copernicus.eu |
| BigEarthNet-SAR | SAR Labels | http://bigearth.net |
| CrisisMMD | Tweets+Images | https://crisisnlp.qcri.org |
| HumAID | Tweet Labels | https://crisisnlp.qcri.org/humaid_data |
| CREDBANK | Credibility | https://figshare.com/articles/CREDBANK-data |

## Project Structure
```
disaster_detection/
├── configs/           # YAML configs for training and pipeline
├── data/              # Raw + processed datasets
├── models/
│   ├── encoders/      # SAR encoder, Text encoder
│   ├── fusion/        # Cross-attention, TAM
│   └── classifier/    # Disaster classification heads
├── pipeline/
│   ├── kafka/         # Kafka producers and consumers
│   └── spark/         # Spark streaming jobs
├── training/          # Training loop, loss, metrics
├── inference/         # TorchServe handler, REST API
├── dashboard/         # FastAPI backend + Leaflet frontend
├── utils/             # Preprocessing utilities
└── scripts/           # Dataset download + preprocessing scripts
```
