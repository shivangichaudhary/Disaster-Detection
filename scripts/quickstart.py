"""
scripts/quickstart.py
─────────────────────
Run the complete pipeline end-to-end in a single script.
Useful for demo / smoke testing without any external dependencies.

Usage:
  python scripts/quickstart.py --mode train    # train on synthetic data
  python scripts/quickstart.py --mode infer    # run inference demo
  python scripts/quickstart.py --mode eval     # full evaluation + ablation
  python scripts/quickstart.py --mode stream   # start Kafka producers + Spark streaming
  python scripts/quickstart.py --mode all      # train → infer → eval (no stream)

Logs are written to  logs/quickstart_<timestamp>.log  automatically.
"""

import sys
import argparse
from pathlib import Path
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import torch.nn.functional as F
from loguru import logger

# ── Ensure logs/ directory exists and capture output there ───────────────────

def _setup_logging(log_dir: Path = _PROJECT_ROOT / "logs") -> Path:
    """Create logs/ directory and add a file sink to loguru."""
    log_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    log_file = log_dir / f"quickstart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(
        str(log_file),
        rotation="50 MB",
        retention="7 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
        encoding="utf-8",
    )
    logger.info(f"Log file: {log_file}")
    return log_file


# ── Quick Train ───────────────────────────────────────────────────────────────

def run_quick_train(epochs: int = 5, batch_size: int = 16, data_mode: str = "synthetic"):
    """Train the fusion model on synthetic or real data."""
    import yaml
    from torch.utils.data import DataLoader, random_split

    from models.encoders.sar_encoder import SAREncoder
    from models.encoders.text_encoder import TextEncoder
    from models.fusion.cross_attention_fusion import DisasterFusionModel
    from training.dataset import DisasterDataset
    from training.train import MultiTaskLoss, get_cosine_schedule_with_warmup

    logger.info(f"=== Quick Training on {data_mode.capitalize()} Data ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Dataset
    if data_mode == "file" and Path("configs/train_config.yaml").exists():
        with open("configs/train_config.yaml") as f:
            cfg = yaml.safe_load(f)["data"]
    else:
        cfg = {"sar_root": "data/processed/sar_patches", "tweet_root": "data/processed/tweets_processed.csv", "num_workers": 0, "pin_memory": False, "split_ratios": [0.7, 0.15, 0.15]}
    dataset = DisasterDataset(cfg, mode=data_mode)
    n = len(dataset)
    train_set, val_set, _ = random_split(dataset, [int(n*0.7), int(n*0.15), n - int(n*0.7) - int(n*0.15)])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=0)
    logger.info(f"Dataset: {len(train_set)} train, {len(val_set)} val")

    # Model
    sar_enc  = SAREncoder(backbone="simple_cnn", output_dim=512)
    txt_enc  = TextEncoder(model_name="simple_bow", output_dim=512)
    config   = {"d_model": 512, "n_heads": 8, "n_layers": 2, "hidden_dim": 256, "dropout": 0.1}
    model    = DisasterFusionModel(sar_enc, txt_enc, config).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    criterion = MultiTaskLoss(binary_weight=1.0, type_weight=1.5, severity_weight=0.8)
    total_steps= len(train_loader) * epochs
    scheduler  = get_cosine_schedule_with_warmup(optimizer, warmup_steps=20, total_steps=total_steps)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            sar       = batch["image"].to(device)
            t_emb     = batch["tweet_embeddings"].to(device)
            t_off     = batch["time_offsets"].to(device)
            t_cred    = batch["credibility"].to(device)
            t_mask    = batch["tweet_mask"].to(device)
            targets   = {
                "binary":   batch["label_binary"].to(device),
                "type":     batch["label_type"].to(device),
                "severity": batch["label_severity"].to(device),
            }
            optimizer.zero_grad()
            out    = model(sar, t_emb, t_off, t_cred, t_mask)
            losses = criterion(out, targets)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += losses["total"].item()

        # Validate
        model.eval()
        val_loss = 0.0
        correct_bin, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                sar   = batch["image"].to(device)
                t_emb = batch["tweet_embeddings"].to(device)
                t_off = batch["time_offsets"].to(device)
                t_cred= batch["credibility"].to(device)
                t_mask= batch["tweet_mask"].to(device)
                targets = {
                    "binary":   batch["label_binary"].to(device),
                    "type":     batch["label_type"].to(device),
                    "severity": batch["label_severity"].to(device),
                }
                out    = model(sar, t_emb, t_off, t_cred, t_mask)
                losses = criterion(out, targets)
                val_loss += losses["total"].item()
                preds    = out["binary"].argmax(dim=-1)
                correct_bin += (preds == targets["binary"]).sum().item()
                total += len(preds)

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        acc       = correct_bin / max(total, 1)
        lr        = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch}/{epochs} | "
            f"Train loss: {avg_train:.4f} | "
            f"Val loss: {avg_val:.4f} | "
            f"Binary acc: {acc:.3f} | "
            f"LR: {lr:.2e}"
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            Path("checkpoints").mkdir(exist_ok=True)
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "val_loss": best_val_loss,
            }, "checkpoints/best_model.pt")
            logger.success(f"  ✓ Saved best model (val_loss={best_val_loss:.4f})")

    logger.success(f"Training complete! Best val loss: {best_val_loss:.4f}")
    return model


# ── Quick Inference ───────────────────────────────────────────────────────────

def run_quick_infer(data_mode: str = "synthetic"):
    """Run single inference without a running server."""
    import yaml
    import numpy as np
    from models.encoders.sar_encoder import SAREncoder
    from models.encoders.text_encoder import TextEncoder
    from models.fusion.cross_attention_fusion import DisasterFusionModel

    logger.info("=== Quick Inference Demo ===")

    device = torch.device("cpu")
    sar_enc  = SAREncoder(backbone="simple_cnn", output_dim=512)
    txt_enc  = TextEncoder(model_name="simple_bow", output_dim=512)
    config   = {"d_model": 512, "n_heads": 8, "n_layers": 2, "hidden_dim": 256, "dropout": 0.0}
    model    = DisasterFusionModel(sar_enc, txt_enc, config).eval()

    if Path("checkpoints/best_model.pt").exists():
        ckpt = torch.load("checkpoints/best_model.pt", map_location=device)
        model.load_state_dict(ckpt["model_state"])
        logger.info("Loaded trained weights.")

    TYPE_NAMES = ["flood", "earthquake", "wildfire", "cyclone", "landslide", "none"]
    SEV_NAMES  = ["low", "medium", "high"]

    test_cases = [
        {
            "desc": "Flood scenario (low SAR backscatter + flood tweets)",
            "tweets": [
                "Severe flooding in Mumbai! Roads completely underwater #flood",
                "Flash floods hit coastal areas, residents evacuating #flooding",
                "Water levels rising fast near Bandra. Emergency services deployed",
            ],
            "sar_type": "flood",
        },
        {
            "desc": "Earthquake scenario",
            "tweets": [
                "Earthquake just hit Delhi! Buildings shaking. 6.2 magnitude reported",
                "Aftershocks continuing in north India #earthquake #tremor",
            ],
            "sar_type": "earthquake",
        },
        {
            "desc": "Normal (no disaster)",
            "tweets": [
                "Beautiful weather in Chennai today. Great for a walk!",
                "New restaurant opened downtown. Amazing food!",
            ],
            "sar_type": "none",
        },
    ]

    from training.dataset import DisasterDataset
    if data_mode == "file" and Path("configs/train_config.yaml").exists():
        with open("configs/train_config.yaml") as f:
            cfg = yaml.safe_load(f)["data"]
    else:
        cfg = {"sar_root": "data/processed/sar_patches", "tweet_root": "data/processed/tweets_processed.csv", "num_workers": 0, "pin_memory": False, "split_ratios": [0.7, 0.15, 0.15]}
    ds = DisasterDataset(cfg, mode=data_mode)

    for i, case in enumerate(test_cases):
        logger.info(f"\nTest case {i+1}: {case['desc']}")

        # SAR patch
        sar_patch = ds._synthetic_sar(case["sar_type"])
        sar_tensor= torch.from_numpy(sar_patch).unsqueeze(0)

        # Tweet embeddings
        T = len(case["tweets"])
        texts = case["tweets"]
        with torch.no_grad():
            t_emb  = txt_enc(texts=texts).unsqueeze(0)  # (1, T, D)
            t_off  = torch.randint(0, 30, (1, T))
            t_cred = torch.tensor([[0.8] * T])
            t_mask = torch.ones(1, T, dtype=torch.bool)

            output = model(sar_tensor, t_emb, t_off, t_cred, t_mask)

        bin_probs  = F.softmax(output["binary"],   dim=-1)[0]
        type_probs = F.softmax(output["type"],     dim=-1)[0]
        sev_probs  = F.softmax(output["severity"], dim=-1)[0]

        pred_type  = TYPE_NAMES[type_probs.argmax()]
        pred_sev   = SEV_NAMES[sev_probs.argmax()]
        confidence = float(bin_probs[1])

        logger.info(f"  Disaster: {'YES' if confidence > 0.5 else 'NO'} "
                    f"(confidence={confidence:.2%})")
        logger.info(f"  Type: {pred_type} | Severity: {pred_sev}")
        logger.info(f"  Type probs: { {t: f'{p:.3f}' for t,p in zip(TYPE_NAMES, type_probs.tolist())} }")


# ── Streaming Mode ────────────────────────────────────────────────────────────

def run_stream(
    source: str   = "mock",
    mode:   str   = "tweet_only",
    tweet_interval: float = 1.5,
    sar_interval:   float = 15.0,
):
    """
    Start the Spark Structured Streaming pipeline with mock data producers.

    Steps:
      1. Start mock tweet producer in a background thread.
      2. If mode='fusion', also start the mock SAR producer.
      3. Launch the Spark streaming job (blocks until Ctrl-C).

    Gracefully degrades:
      • If kafka-python is missing → logs a warning, proceeds anyway.
      • If PySpark is missing      → runs a simple console echo loop instead.
    """
    import time
    import threading
    import signal

    logger.info("=== Streaming Pipeline ===")
    logger.info(f"Source: {source}  |  Mode: {mode}")
    logger.info(f"Tweet interval: {tweet_interval}s  |  SAR interval: {sar_interval}s")

    stop_event = threading.Event()

    # ── Signal handler for clean shutdown ─────────────────────────────────────
    def _handle_sigint(sig, frame):
        logger.info("Ctrl-C received — stopping streaming pipeline...")
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_sigint)

    # ── Tweet producer ────────────────────────────────────────────────────────
    tweet_thread = None
    try:
        from pipeline.kafka.producers import KafkaProducerWrapper, DisasterTwitterProducer
        kafka_prod   = KafkaProducerWrapper()
        tweet_prod   = DisasterTwitterProducer(kafka_prod)

        def _run_tweets():
            logger.info("[Producer] Mock tweet producer started → tweets-raw")
            try:
                tweet_prod._run_mock(interval=tweet_interval)
            except Exception as exc:
                if not stop_event.is_set():
                    logger.error(f"[Producer] Tweet producer error: {exc}")

        tweet_thread = threading.Thread(target=_run_tweets, daemon=True, name="tweet-producer")
        tweet_thread.start()
        logger.info("[Producer] Tweet producer thread started ✓")
    except Exception as exc:
        logger.warning(f"[Producer] Could not start tweet producer: {exc}")

    # ── SAR producer (fusion mode only) ───────────────────────────────────────
    sar_thread = None
    if mode == "fusion":
        try:
            from pipeline.kafka.producers import KafkaProducerWrapper
            from pipeline.kafka.sar_mock_producer import SARMockProducer
            sar_kafka = KafkaProducerWrapper()
            sar_prod  = SARMockProducer(sar_kafka, interval=sar_interval, disaster_ratio=0.70)

            def _run_sar():
                logger.info("[SAR Producer] Mock SAR producer started → sar-raw")
                try:
                    sar_prod.start()
                except Exception as exc:
                    if not stop_event.is_set():
                        logger.error(f"[SAR Producer] Error: {exc}")

            sar_thread = threading.Thread(target=_run_sar, daemon=True, name="sar-producer")
            sar_thread.start()
            logger.info("[SAR Producer] SAR producer thread started ✓")
        except Exception as exc:
            logger.warning(f"[SAR Producer] Could not start: {exc}")

    # Give producers a moment to warm up
    time.sleep(2)

    # ── Spark streaming job ───────────────────────────────────────────────────
    try:
        from pipeline.spark.streaming_job import run_streaming_pipeline
        logger.info("[Stream] Starting Spark streaming pipeline (Ctrl-C to stop)...")
        run_streaming_pipeline(
            mode=mode,
            enable_kafka_sink=True,
            enable_postgres_sink=False,
            enable_console_sink=True,
            await_termination=True,
        )
    except ImportError:
        # PySpark not available — fall back to a simple console echo loop
        logger.warning(
            "[Stream] PySpark not installed — running console-only echo loop.\n"
            "         Install PySpark with:  pip install pyspark>=3.4.0"
        )
        logger.info("[Stream] Press Ctrl-C to stop.")
        while not stop_event.is_set():
            time.sleep(1)
    except Exception as exc:
        logger.error(f"[Stream] Streaming job error: {exc}")
    finally:
        stop_event.set()
        logger.info("[Stream] Streaming pipeline stopped.")


# ── Ablation Study ────────────────────────────────────────────────────────────

def run_ablation(data_mode: str = "synthetic"):
    """
    Compare SAR-only vs Tweet-only vs Fused model performance.
    This is a key result for the research paper.
    """
    import yaml
    import numpy as np
    from training.dataset import DisasterDataset
    from models.encoders.sar_encoder import SAREncoder
    from models.encoders.text_encoder import TextEncoder
    from models.fusion.cross_attention_fusion import DisasterFusionModel

    logger.info("=== Ablation Study: SAR-only vs Tweet-only vs Fused ===")

    device = torch.device("cpu")
    if data_mode == "file" and Path("configs/train_config.yaml").exists():
        with open("configs/train_config.yaml") as f:
            cfg = yaml.safe_load(f)["data"]
    else:
        cfg = {"sar_root": "data/processed/sar_patches", "tweet_root": "data/processed/tweets_processed.csv", "num_workers": 0, "pin_memory": False, "split_ratios": [0.7, 0.15, 0.15]}
    ds     = DisasterDataset(cfg, mode=data_mode)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)

    sar_enc = SAREncoder(backbone="simple_cnn", output_dim=512)
    txt_enc = TextEncoder(model_name="simple_bow", output_dim=512)
    config  = {"d_model": 512, "n_heads": 8, "n_layers": 2, "hidden_dim": 256, "dropout": 0.0}
    model   = DisasterFusionModel(sar_enc, txt_enc, config).eval()

    results = {}
    conditions = ["fused", "sar_only", "tweet_only"]

    for cond in conditions:
        correct, total = 0, 0
        with torch.no_grad():
            for batch in loader:
                sar   = batch["image"].to(device)
                t_emb = batch["tweet_embeddings"].to(device)
                t_off = batch["time_offsets"].to(device)
                t_cred= batch["credibility"].to(device)
                t_mask= batch["tweet_mask"].to(device)
                labels= batch["label_binary"].to(device)

                if cond == "sar_only":
                    # Zero out tweet embeddings
                    t_emb  = torch.zeros_like(t_emb)
                    t_cred = torch.zeros_like(t_cred)
                elif cond == "tweet_only":
                    # Zero out SAR image
                    sar = torch.zeros_like(sar)

                out   = model(sar, t_emb, t_off, t_cred, t_mask)
                preds = out["binary"].argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total   += len(labels)

        acc = correct / max(total, 1)
        results[cond] = acc
        logger.info(f"  {cond:12s} accuracy: {acc:.3f} ({correct}/{total})")

    logger.info("\n  ┌─────────────────────────────────┐")
    logger.info("  │      Ablation Study Results      │")
    logger.info("  ├──────────────┬──────────────────┤")
    for cond, acc in results.items():
        bar = "█" * int(acc * 20)
        logger.info(f"  │ {cond:12s} │ {bar:<20s} {acc:.3f} │")
    logger.info("  └──────────────┴──────────────────┘")

    improvement = results.get("fused", 0) - max(
        results.get("sar_only", 0), results.get("tweet_only", 0)
    )
    logger.info(f"\n  Fusion improvement over best single modality: {improvement:+.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Disaster Detection Quick Start",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/quickstart.py --mode train\n"
            "  python scripts/quickstart.py --mode stream\n"
            "  python scripts/quickstart.py --mode stream --stream_mode fusion\n"
            "  python scripts/quickstart.py --mode all\n"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["train", "infer", "eval", "stream", "all"],
        default="all",
        help=(
            "train   – train on synthetic data\n"
            "infer   – run inference demo\n"
            "eval    – ablation study (SAR-only vs tweet-only vs fused)\n"
            "stream  – start Kafka mock producers + Spark streaming pipeline\n"
            "all     – train → infer → eval  (stream is NOT included in 'all')"
        ),
    )
    parser.add_argument(
        "--data_mode",
        choices=["file", "synthetic"],
        default="synthetic",
        help="Dataset source for train/infer/eval modes (default: synthetic)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Training epochs (default: 5)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Training batch size (default: 16)",
    )
    # ── Streaming-specific args ───────────────────────────────────────────────
    parser.add_argument(
        "--stream_source",
        choices=["mock", "bluesky", "mastodon", "twitter"],
        default="mock",
        help="Social media source for --mode stream (default: mock)",
    )
    parser.add_argument(
        "--stream_mode",
        choices=["tweet_only", "fusion"],
        default="tweet_only",
        help="'tweet_only' or 'fusion' (SAR + tweets) for --mode stream (default: tweet_only)",
    )
    parser.add_argument(
        "--tweet_interval",
        type=float,
        default=1.5,
        help="Seconds between mock tweet events (default: 1.5)",
    )
    parser.add_argument(
        "--sar_interval",
        type=float,
        default=15.0,
        help="Seconds between mock SAR events in fusion mode (default: 15.0)",
    )
    args = parser.parse_args()

    # Always set up logging first so every mode gets a log file
    _setup_logging()

    if args.mode == "stream":
        run_stream(
            source=args.stream_source,
            mode=args.stream_mode,
            tweet_interval=args.tweet_interval,
            sar_interval=args.sar_interval,
        )
    else:
        if args.mode in ("train", "all"):
            run_quick_train(
                epochs=args.epochs,
                batch_size=args.batch_size,
                data_mode=args.data_mode,
            )

        if args.mode in ("infer", "all"):
            run_quick_infer(data_mode=args.data_mode)

        if args.mode in ("eval", "all"):
            run_ablation(data_mode=args.data_mode)

        logger.success("Quickstart complete!")
