"""
training/evaluate.py
─────────────────────
Full evaluation suite:
  • Per-class F1, Precision, Recall
  • Confusion matrix
  • Ablation study (SAR-only vs Tweet-only vs Fused)
  • Latency benchmark
  • Saves results as CSV + plots
"""

import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.encoders.sar_encoder  import SAREncoder
from models.encoders.text_encoder import TextEncoder
from models.fusion.cross_attention_fusion import DisasterFusionModel
from training.dataset import DisasterDataset, LABEL_TO_TYPE

TYPE_NAMES = ["flood", "earthquake", "wildfire", "cyclone", "landslide", "none"]
SEV_NAMES  = ["low", "medium", "high"]


def load_model(model_path: str, device: torch.device) -> DisasterFusionModel:
    sar_enc = SAREncoder(backbone="simple_cnn", output_dim=512)
    txt_enc = TextEncoder(model_name="simple_bow", output_dim=512)
    config  = {"d_model": 512, "n_heads": 8, "n_layers": 2, "hidden_dim": 256, "dropout": 0.0}
    model   = DisasterFusionModel(sar_enc, txt_enc, config)

    if Path(model_path).exists():
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        logger.info(f"Loaded model from {model_path} (epoch {ckpt.get('epoch','?')})")
    else:
        logger.warning(f"No checkpoint at {model_path}. Using untrained model.")

    return model.to(device).eval()


@torch.no_grad()
def run_evaluation(
    model: DisasterFusionModel,
    loader: DataLoader,
    device: torch.device,
    condition: str = "fused",
) -> dict:
    """
    Evaluate model on a DataLoader.
    condition: 'fused' | 'sar_only' | 'tweet_only'
    """
    all_bin_true, all_bin_pred, all_bin_proba = [], [], []
    all_type_true, all_type_pred = [], []
    all_sev_true,  all_sev_pred  = [], []
    latencies = []

    for batch in loader:
        sar    = batch["image"].to(device)
        t_emb  = batch["tweet_embeddings"].to(device)
        t_off  = batch["time_offsets"].to(device)
        t_cred = batch["credibility"].to(device)
        t_mask = batch["tweet_mask"].to(device)

        if condition == "sar_only":
            t_emb  = torch.zeros_like(t_emb)
            t_cred = torch.zeros_like(t_cred)
        elif condition == "tweet_only":
            sar = torch.zeros_like(sar)

        t0  = time.perf_counter()
        out = model(sar, t_emb, t_off, t_cred, t_mask)
        latencies.append((time.perf_counter() - t0) * 1000 / sar.shape[0])

        bin_proba = F.softmax(out["binary"],   dim=-1)[:, 1].cpu()
        bin_pred  = out["binary"].argmax(dim=-1).cpu()
        type_pred = out["type"].argmax(dim=-1).cpu()
        sev_pred  = out["severity"].argmax(dim=-1).cpu()

        all_bin_true.extend(batch["label_binary"].tolist())
        all_bin_pred.extend(bin_pred.tolist())
        all_bin_proba.extend(bin_proba.tolist())
        all_type_true.extend(batch["label_type"].tolist())
        all_type_pred.extend(type_pred.tolist())
        all_sev_true.extend(batch["label_severity"].tolist())
        all_sev_pred.extend(sev_pred.tolist())

    # ── Metrics ──────────────────────────────────────────────────────────────
    from sklearn.metrics import (
        f1_score, precision_score, recall_score,
        accuracy_score, confusion_matrix, roc_auc_score, classification_report
    )

    bin_true = np.array(all_bin_true)
    bin_pred_arr = np.array(all_bin_pred)
    type_true= np.array(all_type_true)
    type_pred_arr= np.array(all_type_pred)
    sev_true = np.array(all_sev_true)
    sev_pred_arr = np.array(all_sev_pred)

    results = {
        "condition": condition,
        # Binary
        "bin_accuracy":  accuracy_score(bin_true, bin_pred_arr),
        "bin_f1":        f1_score(bin_true, bin_pred_arr, average="binary", zero_division=0),
        "bin_precision": precision_score(bin_true, bin_pred_arr, average="binary", zero_division=0),
        "bin_recall":    recall_score(bin_true, bin_pred_arr, average="binary", zero_division=0),
        # Type
        "type_accuracy": accuracy_score(type_true, type_pred_arr),
        "type_f1_macro": f1_score(type_true, type_pred_arr, average="macro", zero_division=0),
        "type_f1_weighted": f1_score(type_true, type_pred_arr, average="weighted", zero_division=0),
        # Severity
        "sev_accuracy":  accuracy_score(sev_true, sev_pred_arr),
        "sev_f1_macro":  f1_score(sev_true, sev_pred_arr, average="macro", zero_division=0),
        # Latency
        "latency_mean_ms":  float(np.mean(latencies)),
        "latency_p95_ms":   float(np.percentile(latencies, 95)),
        # Per-class F1
        "per_class_f1": dict(zip(
            TYPE_NAMES,
            f1_score(type_true, type_pred_arr, average=None, zero_division=0, labels=list(range(6))).tolist()
        )),
    }

    # Try AUROC (needs both classes present)
    try:
        results["bin_auroc"] = roc_auc_score(bin_true, np.array(all_bin_proba))
    except Exception:
        results["bin_auroc"] = 0.0

    return results


def print_results_table(all_results: list):
    """Print a formatted comparison table."""
    print("\n" + "═"*70)
    print("  EVALUATION RESULTS")
    print("═"*70)
    print(f"  {'Metric':<28} {'SAR-only':>10} {'Tweet-only':>12} {'Fused':>10}")
    print("─"*70)

    metrics = [
        ("Binary Accuracy",    "bin_accuracy"),
        ("Binary F1",          "bin_f1"),
        ("Binary Precision",   "bin_precision"),
        ("Binary Recall",      "bin_recall"),
        ("Binary AUROC",       "bin_auroc"),
        ("Type F1 (macro)",    "type_f1_macro"),
        ("Type F1 (weighted)", "type_f1_weighted"),
        ("Severity F1",        "sev_f1_macro"),
        ("Latency (ms/sample)","latency_mean_ms"),
    ]

    result_by_cond = {r["condition"]: r for r in all_results}

    for label, key in metrics:
        vals = []
        for cond in ["sar_only", "tweet_only", "fused"]:
            v = result_by_cond.get(cond, {}).get(key, 0.0)
            vals.append(f"{v:.3f}")
        print(f"  {label:<28} {vals[0]:>10} {vals[1]:>12} {vals[2]:>10}")

    print("─"*70)

    # Per-class F1 (fused only)
    if "fused" in result_by_cond:
        print("\n  Per-class F1 (Fused model):")
        per_cls = result_by_cond["fused"]["per_class_f1"]
        for cls, f1 in per_cls.items():
            bar = "█" * int(f1 * 20)
            print(f"  {cls:<12} {bar:<20} {f1:.3f}")

    print("═"*70 + "\n")


def save_results(all_results: list, out_dir: str = "results"):
    """Save evaluation results to CSV and JSON."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Main metrics
    rows = []
    for r in all_results:
        row = {k: v for k, v in r.items() if k != "per_class_f1"}
        rows.append(row)
    pd.DataFrame(rows).to_csv(f"{out_dir}/evaluation_results.csv", index=False)

    # Per-class F1
    per_class = {}
    for r in all_results:
        per_class[r["condition"]] = r.get("per_class_f1", {})
    pd.DataFrame(per_class).to_csv(f"{out_dir}/per_class_f1.csv")

    # Full JSON
    with open(f"{out_dir}/evaluation_full.json", "w") as f:
        json.dump(all_results, f, indent=2)

    logger.success(f"Results saved to {out_dir}/")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Evaluating on: {device}")

    # Load config file if it exists
    import yaml
    if Path(args.config).exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f)["data"]
    else:
        cfg = {"sar_root": "data/processed/sar_patches", "tweet_root": "data/processed/tweets_processed.csv", "num_workers": 0,
               "pin_memory": False, "split_ratios": [0.7, 0.15, 0.15]}

    data_mode = args.data_mode or cfg.get("mode", "synthetic")
    logger.info(f"Using dataset mode: {data_mode}")
    dataset = DisasterDataset(cfg, mode=data_mode)
    n       = len(dataset)
    _, _, test_set = random_split(dataset, [int(n*0.7), int(n*0.15), n - int(n*0.7) - int(n*0.15)],
                                  generator=torch.Generator().manual_seed(42))
    loader  = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)
    logger.info(f"Test set: {len(test_set)} samples")

    # Model
    model = load_model(args.model_path, device)

    # Run ablation
    all_results = []
    for condition in ["fused", "sar_only", "tweet_only"]:
        logger.info(f"Evaluating: {condition}...")
        results = run_evaluation(model, loader, device, condition)
        all_results.append(results)
        logger.info(f"  {condition}: Binary F1={results['bin_f1']:.3f} | "
                    f"Type F1={results['type_f1_macro']:.3f} | "
                    f"Latency={results['latency_mean_ms']:.1f}ms")

    print_results_table(all_results)
    save_results(all_results, out_dir=args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate disaster detection model")
    parser.add_argument("--model_path", default="checkpoints/best_model.pt")
    parser.add_argument("--config",     default="configs/train_config.yaml")
    parser.add_argument("--data_mode",  default=None, choices=["file", "synthetic"])
    parser.add_argument("--out_dir",    default="results")
    args = parser.parse_args()
    main(args)
