"""
training/train.py
─────────────────
Full training loop for DisasterFusionModel.
  • Multi-task loss (binary + type + severity + contrastive)
  • Focal loss for class imbalance
  • Mixed precision training
  • Cosine LR schedule with warmup
  • W&B / tensorboard logging
  • Checkpoint saving (best F1)
"""

import os
import sys
import yaml
import argparse
import random
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.amp import GradScaler, autocast
from loguru import logger

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.encoders.sar_encoder import SAREncoder
from models.encoders.text_encoder import TextEncoder
from models.fusion.cross_attention_fusion import DisasterFusionModel
from training.dataset import DisasterDataset, DisasterMetrics


# ── Loss Functions ────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in disaster detection.
    FL(p) = -α(1-p)^γ log(p)
    Reduces weight of easy examples, focuses on hard misclassified ones.
    Reference: Lin et al. (2017) Focal Loss for Dense Object Detection.
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt      = torch.exp(-ce_loss)
        focal   = ((1 - pt) ** self.gamma) * ce_loss
        return focal.mean()


class MultiTaskLoss(nn.Module):
    """
    Weighted sum of:
      1. Focal CE for binary classification
      2. Focal CE for disaster type classification
      3. Focal CE for severity classification
      4. Contrastive loss (InfoNCE) for cross-modal alignment
    """

    def __init__(
        self,
        binary_weight:     float = 1.0,
        type_weight:       float = 1.5,
        severity_weight:   float = 0.8,
        contrastive_weight:float = 0.5,
        focal_gamma:       float = 2.0,
        num_types:         int = 6,
    ):
        super().__init__()
        self.w_binary      = binary_weight
        self.w_type        = type_weight
        self.w_severity    = severity_weight
        self.w_contrastive = contrastive_weight

        self.focal_binary   = FocalLoss(gamma=focal_gamma)
        self.focal_type     = FocalLoss(gamma=focal_gamma)
        self.focal_severity = FocalLoss(gamma=focal_gamma)

    def forward(
        self,
        outputs: dict,
        targets: dict,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: model output dict with 'binary', 'type', 'severity', 'h_img', 'h_txt'
            targets: dict with 'binary', 'type', 'severity' label tensors
        """
        loss_binary   = self.focal_binary(outputs["binary"],   targets["binary"])
        loss_type     = self.focal_type(outputs["type"],       targets["type"])
        loss_severity = self.focal_severity(outputs["severity"], targets["severity"])

        # Contrastive loss (from fusion module)
        loss_contrastive = torch.tensor(0.0, device=outputs["h_img"].device)
        if "h_img" in outputs and "h_txt" in outputs:
            # Only compute on disaster samples (binary label = 1)
            is_disaster = (targets["binary"] == 1)
            if is_disaster.sum() > 1:
                loss_contrastive = F.mse_loss(
                    outputs["h_img"][is_disaster],
                    outputs["h_txt"][is_disaster].detach(),
                )

        total = (
            self.w_binary      * loss_binary +
            self.w_type        * loss_type +
            self.w_severity    * loss_severity +
            self.w_contrastive * loss_contrastive
        )

        return {
            "total":       total,
            "binary":      loss_binary,
            "type":        loss_type,
            "severity":    loss_severity,
            "contrastive": loss_contrastive,
        }


# ── LR Scheduler ─────────────────────────────────────────────────────────────

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.01,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup → cosine decay."""
    import math

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(
            min_lr_ratio,
            0.5 * (1.0 + math.cos(math.pi * progress)),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Training Step ─────────────────────────────────────────────────────────────

def train_epoch(
    model: DisasterFusionModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: MultiTaskLoss,
    scaler: GradScaler,
    device: torch.device,
    grad_accum: int = 1,
    scheduler=None,
) -> Dict[str, float]:
    model.train()
    total_losses = defaultdict(float)
    n_batches    = 0

    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        sar       = batch["image"].to(device)
        tweet_emb = batch["tweet_embeddings"].to(device)
        offsets   = batch["time_offsets"].to(device)
        cred      = batch["credibility"].to(device)
        mask      = batch["tweet_mask"].to(device)
        targets   = {
            "binary":   batch["label_binary"].to(device),
            "type":     batch["label_type"].to(device),
            "severity": batch["label_severity"].to(device),
        }

        with autocast("cuda", enabled=scaler is not None):
            outputs = model(sar, tweet_emb, offsets, cred, mask)
            losses  = criterion(outputs, targets)
            loss    = losses["total"] / grad_accum

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum == 0:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            optimizer.zero_grad()
            if scheduler:
                scheduler.step()

        for k, v in losses.items():
            total_losses[k] += v.item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in total_losses.items()}


@torch.no_grad()
def validate(
    model: DisasterFusionModel,
    loader: DataLoader,
    criterion: MultiTaskLoss,
    metrics: "DisasterMetrics",
    device: torch.device,
) -> Dict:
    model.eval()
    total_losses = defaultdict(float)
    n_batches = 0

    for batch in loader:
        sar       = batch["image"].to(device)
        tweet_emb = batch["tweet_embeddings"].to(device)
        offsets   = batch["time_offsets"].to(device)
        cred      = batch["credibility"].to(device)
        mask      = batch["tweet_mask"].to(device)
        targets   = {
            "binary":   batch["label_binary"].to(device),
            "type":     batch["label_type"].to(device),
            "severity": batch["label_severity"].to(device),
        }

        outputs = model(sar, tweet_emb, offsets, cred, mask)
        losses  = criterion(outputs, targets)

        metrics.update(outputs, targets)
        for k, v in losses.items():
            total_losses[k] += v.item()
        n_batches += 1

    val_losses  = {k: v / max(n_batches, 1) for k, v in total_losses.items()}
    val_metrics = metrics.compute()
    metrics.reset()

    return {**val_losses, **val_metrics}


# ── Main Training Loop ────────────────────────────────────────────────────────

def train(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Reproducibility
    seed = cfg["project"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")

    # ── Build Dataset ────────────────────────────────────────────────────────
    dataset = DisasterDataset(cfg["data"], mode=cfg["data"].get("mode", "synthetic"))
    n       = len(dataset)
    splits  = cfg["data"]["split_ratios"]
    n_train = int(n * splits[0])
    n_val   = int(n * splits[1])
    n_test  = n - n_train - n_val

    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(
        train_set, batch_size=cfg["training"]["batch_size"],
        shuffle=True, num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg["training"]["batch_size"],
        shuffle=False, num_workers=cfg["data"]["num_workers"],
    )

    logger.info(f"Dataset: {n_train} train | {n_val} val | {n_test} test")

    # ── Build Model ──────────────────────────────────────────────────────────
    sar_cfg  = cfg["model"]["sar_encoder"]
    txt_cfg  = cfg["model"]["text_encoder"]
    fus_cfg  = {**cfg["model"]["cross_attention"], **cfg["model"]["classifier"]}

    sar_encoder  = SAREncoder(
        backbone=sar_cfg["backbone"],
        pretrained=sar_cfg["pretrained"],
        output_dim=sar_cfg["output_dim"],
        freeze_layers=sar_cfg["freeze_layers"],
    )
    text_encoder = TextEncoder(
        model_name=txt_cfg["model_name"],
        output_dim=txt_cfg["output_dim"],
        freeze_layers=txt_cfg["freeze_layers"],
    )
    model = DisasterFusionModel(sar_encoder, text_encoder, fus_cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_params:,}")

    # ── Optimiser & Scheduler ────────────────────────────────────────────────
    train_cfg = cfg["training"]
    opt_cfg   = train_cfg["optimizer"]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg["weight_decay"],
        betas=tuple(opt_cfg["betas"]),
    )

    total_steps   = len(train_loader) * train_cfg["epochs"]
    warmup_steps  = len(train_loader) * cfg["training"]["scheduler"]["warmup_epochs"]
    scheduler     = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    criterion = MultiTaskLoss(
        binary_weight=train_cfg["loss"]["binary_weight"],
        type_weight=train_cfg["loss"]["type_weight"],
        severity_weight=train_cfg["loss"]["severity_weight"],
        focal_gamma=train_cfg["loss"]["focal_gamma"],
    )

    scaler  = GradScaler("cuda") if train_cfg.get("mixed_precision") and device.type == "cuda" else None
    metrics = DisasterMetrics(device=device)

    # ── Training ─────────────────────────────────────────────────────────────
    ckpt_dir = Path(cfg["logging"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_f1       = 0.0
    patience_ct   = 0
    patience      = train_cfg["early_stopping"]["patience"]
    watch_metric  = train_cfg["early_stopping"]["metric"]

    for epoch in range(1, train_cfg["epochs"] + 1):
        logger.info(f"Epoch {epoch}/{train_cfg['epochs']}")

        train_losses = train_epoch(
            model, train_loader, optimizer, criterion, scaler, device,
            grad_accum=train_cfg["grad_accum_steps"], scheduler=scheduler,
        )
        val_results = validate(model, val_loader, criterion, metrics, device)

        # Log
        lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"  Train loss: {train_losses['total']:.4f} "
            f"(bin={train_losses['binary']:.3f}, type={train_losses['type']:.3f})"
        )
        logger.info(
            f"  Val   loss: {val_results['total']:.4f} | "
            f"F1_macro={val_results.get('f1_macro', 0):.4f} | "
            f"F1_binary={val_results.get('f1_binary', 0):.4f} | "
            f"LR={lr:.2e}"
        )

        # Save checkpoint
        current_metric = val_results.get(watch_metric.replace("val_", ""), 0)
        if current_metric > best_f1:
            best_f1 = current_metric
            patience_ct = 0
            ckpt_path = ckpt_dir / "best_model.pt"
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "opt_state":   optimizer.state_dict(),
                "best_f1":     best_f1,
                "config":      cfg,
            }, ckpt_path)
            logger.success(f"  ✓ Saved best model (F1={best_f1:.4f}) → {ckpt_path}")
        else:
            patience_ct += 1
            if patience_ct >= patience:
                logger.warning(f"Early stopping at epoch {epoch}")
                break

    logger.success(f"Training complete. Best F1: {best_f1:.4f}")
    return best_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()
    train(args.config)
