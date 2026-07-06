"""
models/encoders/sar_encoder.py
──────────────────────────────
SAR image encoder supporting:
  • ViT-B/16  (Vision Transformer) — default, better for global context
  • ResNet-50 — lighter, faster training
Both output 512-dim feature vectors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from loguru import logger

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    logger.warning("timm not installed. Using fallback CNN encoder.")


class SAREncoder(nn.Module):
    """
    Encodes (B, 2, 256, 256) SAR patches → (B, output_dim) feature vectors.

    Args:
        backbone:     'vit_base_patch16_224' | 'resnet50' | 'simple_cnn'
        pretrained:   use ImageNet pretrained weights (fine-tune for SAR)
        output_dim:   feature dimension (512 recommended)
        freeze_layers: number of backbone layers to freeze
        in_channels:  2 for SAR (VV + VH bands)
    """

    def __init__(
        self,
        backbone: str = "vit_base_patch16_224",
        pretrained: bool = True,
        output_dim: int = 512,
        freeze_layers: int = 6,
        in_channels: int = 2,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.output_dim    = output_dim

        if TIMM_AVAILABLE and backbone != "simple_cnn":
            self.backbone, feat_dim = self._build_timm_backbone(
                backbone, pretrained, in_channels, freeze_layers
            )
        else:
            self.backbone, feat_dim = self._build_simple_cnn(in_channels)

        # Projection head: backbone_dim → output_dim
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )

    def _build_timm_backbone(
        self, name: str, pretrained: bool, in_channels: int, freeze_layers: int
    ):
        """Build timm backbone and adapt for SAR input channels."""
        model = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=0,          # remove classification head
            in_chans=in_channels,   # adapt from 3-channel to 2-channel
        )
        feat_dim = model.num_features

        # Freeze early layers for transfer learning
        if freeze_layers > 0:
            params = list(model.parameters())
            for p in params[:freeze_layers * 10]:
                p.requires_grad = False

        logger.info(
            f"SAREncoder: {name} | feat_dim={feat_dim} | "
            f"pretrained={pretrained} | frozen_layers={freeze_layers}"
        )
        return model, feat_dim

    def _build_simple_cnn(self, in_channels: int):
        """
        Lightweight CNN for SAR encoding when timm is unavailable.
        4 conv blocks with residual connections → global average pooling.
        """
        logger.info("SAREncoder: Using simple CNN backbone (timm not available)")

        class SimpleSARCNN(nn.Module):
            def __init__(self, in_ch):
                super().__init__()
                self.stem = nn.Sequential(
                    nn.Conv2d(in_ch, 64, 7, stride=2, padding=3, bias=False),
                    nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                    nn.MaxPool2d(3, stride=2, padding=1),
                )
                self.layer1 = self._res_block(64, 128)
                self.layer2 = self._res_block(128, 256, stride=2)
                self.layer3 = self._res_block(256, 512, stride=2)
                self.layer4 = self._res_block(512, 1024, stride=2)
                self.gap     = nn.AdaptiveAvgPool2d(1)
                self.num_features = 1024

            def _res_block(self, in_c, out_c, stride=1):
                return nn.Sequential(
                    nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
                    nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
                )

            def forward(self, x):
                x = self.stem(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.gap(x).flatten(1)
                return x

        model = SimpleSARCNN(in_channels)
        return model, 1024

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2, 256, 256) — SAR image tensor (VV, VH bands)
        Returns:
            features: (B, output_dim) — L2-normalised feature vector
        """
        # ViT expects (B, C, 224, 224) — resize if needed
        if "vit" in self.backbone_name and x.shape[-1] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        feats = self.backbone(x)            # (B, feat_dim)

        if feats.dim() > 2:
            feats = feats.mean(dim=list(range(1, feats.dim() - 1)))  # GAP

        projected = self.projector(feats)   # (B, output_dim)
        return F.normalize(projected, dim=-1)


class SAREncoderWithAuxHead(SAREncoder):
    """
    SAR encoder with an auxiliary classification head.
    Used to pretrain the encoder on BigEarthNet labels before fusion training.
    """

    def __init__(self, num_classes: int = 6, **kwargs):
        super().__init__(**kwargs)
        self.aux_classifier = nn.Sequential(
            nn.Linear(self.output_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        feats = super().forward(x)
        if return_aux:
            return feats, self.aux_classifier(feats)
        return feats


if __name__ == "__main__":
    # Test both backbones
    x = torch.randn(4, 2, 256, 256)

    # Test simple CNN (always available)
    enc = SAREncoder(backbone="simple_cnn", output_dim=512)
    out = enc(x)
    print(f"SimpleCNN output: {out.shape}, norm: {out.norm(dim=-1).mean():.4f}")
    assert out.shape == (4, 512), f"Expected (4, 512), got {out.shape}"

    if TIMM_AVAILABLE:
        enc_vit = SAREncoder(
            backbone="vit_base_patch16_224",
            pretrained=False,
            output_dim=512,
            freeze_layers=0,
        )
        out_vit = enc_vit(x)
        print(f"ViT output: {out_vit.shape}, norm: {out_vit.norm(dim=-1).mean():.4f}")

    print("SAREncoder tests passed.")
