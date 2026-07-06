"""
models/fusion/cross_attention_fusion.py
────────────────────────────────────────
Core fusion architecture:
  1. TemporalAlignmentModule  (TAM)  — align tweet windows to SAR timestamps
  2. CrossModalAttention             — bidirectional cross-attention
  3. DisasterFusionModel             — full end-to-end model
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# 1. Temporal Alignment Module (TAM)
# ─────────────────────────────────────────────────────────────────────────────

class TemporalPositionEncoding(nn.Module):
    """
    Sinusoidal temporal position encoding.
    Encodes elapsed minutes relative to SAR acquisition time.
    """
    def __init__(self, d_model: int = 512, max_minutes: int = 720):
        super().__init__()
        pe = torch.zeros(max_minutes, d_model)
        position = torch.arange(0, max_minutes).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, time_offsets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time_offsets: (B, T) — minutes from SAR acquisition, integer
        Returns:
            pos_enc: (B, T, d_model)
        """
        offsets_clamped = time_offsets.long().clamp(0, self.pe.shape[0] - 1)
        return self.pe[offsets_clamped]  # (B, T, d_model)


class TemporalAlignmentModule(nn.Module):
    """
    Aligns asynchronous tweet streams to SAR acquisition timestamps.

    Key idea:
        - SAR captures a snapshot at time t_sar
        - Tweets arrive continuously before/after t_sar
        - TAM uses temporal attention to weight tweets by
          (a) proximity to t_sar  (b) credibility  (c) content relevance

    Input:
        tweet_embeddings: (B, T, D) — up to T tweets in the window
        time_offsets:     (B, T)    — minutes before/after SAR acquisition
        credibility:      (B, T)    — credibility scores [0, 1]
        tweet_mask:       (B, T)    — True for valid tweets

    Output:
        aligned_repr: (B, D) — aligned window representation
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_window_minutes: int = 60,
    ):
        super().__init__()
        self.d_model = d_model

        # Temporal position encoding
        self.temporal_pe = TemporalPositionEncoding(d_model, max_minutes=max_window_minutes * 2)

        # Credibility projection (scalar → d_model)
        self.cred_proj = nn.Linear(1, d_model)

        # Self-attention over tweets in window
        self.tweet_self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )

        # Temporal decay attention
        # Weight tweets closer to SAR timestamp higher
        self.temporal_gate = nn.Sequential(
            nn.Linear(d_model + d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )

        self.layer_norm  = nn.LayerNorm(d_model)
        self.dropout     = nn.Dropout(dropout)

        # Final projection
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        tweet_embeddings: torch.Tensor,          # (B, T, D)
        time_offsets: torch.Tensor,              # (B, T) — minutes offset from SAR
        credibility: torch.Tensor,               # (B, T)
        tweet_mask: Optional[torch.Tensor] = None,  # (B, T) bool
    ) -> torch.Tensor:

        B, T, D = tweet_embeddings.shape

        # 1. Add temporal position encoding
        pos_enc = self.temporal_pe(time_offsets.abs())  # (B, T, D)

        # 2. Add credibility as additive signal
        cred_enc = self.cred_proj(credibility.unsqueeze(-1))  # (B, T, D)

        # Combine: tweet content + temporal position + credibility
        x = tweet_embeddings + pos_enc + cred_enc  # (B, T, D)
        x = self.layer_norm(x)

        # 3. Self-attention over all tweets in window
        key_padding_mask = ~tweet_mask if tweet_mask is not None else None
        attn_out, _ = self.tweet_self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.layer_norm(x + self.dropout(attn_out))  # (B, T, D)

        # 4. Temporal gate: compute attention weight for each tweet
        # Tweets closer in time to SAR acquisition get higher weight
        time_decay = torch.exp(-time_offsets.abs().float() / 30.0).unsqueeze(-1)  # (B, T, 1)

        gate_input = torch.cat([x, pos_enc], dim=-1)  # (B, T, 2D)
        gate_scores= self.temporal_gate(gate_input)   # (B, T, 1)

        # Combine temporal decay with learned gate
        combined_weight = gate_scores + torch.log(time_decay + 1e-8)

        if tweet_mask is not None:
            # Prevent all-False masks from producing NaN in softmax by falling back to uniform attention if empty
            any_valid = tweet_mask.any(dim=1, keepdim=True)  # (B, 1)
            safe_mask = torch.where(any_valid, tweet_mask, torch.ones_like(tweet_mask))
            combined_weight = combined_weight.masked_fill(~safe_mask.unsqueeze(-1), float("-inf"))

        attn_weights = torch.softmax(combined_weight, dim=1)  # (B, T, 1)

        # 5. Weighted aggregation
        aligned = (x * attn_weights).sum(dim=1)  # (B, D)
        aligned = self.out_proj(aligned)
        return F.normalize(aligned, dim=-1)       # (B, D)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cross-Modal Attention Block
# ─────────────────────────────────────────────────────────────────────────────

class CrossModalAttentionBlock(nn.Module):
    """
    One block of bidirectional cross-modal attention.

        SAR features ─── Q ──► attend to Tweet keys/values ──► h_img
        Tweet features ─ Q ──► attend to SAR keys/values ───► h_txt

    Both modalities update each other, enabling joint reasoning.
    """

    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        # SAR queries tweet features
        self.img_to_txt_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        # Tweet queries SAR features
        self.txt_to_img_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # Feed-forward networks
        self.ffn_img = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.ffn_txt = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

        self.norm_img1 = nn.LayerNorm(d_model)
        self.norm_img2 = nn.LayerNorm(d_model)
        self.norm_txt1 = nn.LayerNorm(d_model)
        self.norm_txt2 = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(dropout)

    def forward(
        self,
        v_img: torch.Tensor,   # (B, 1, D) or (B, D) — SAR features
        v_txt: torch.Tensor,   # (B, 1, D) or (B, D) — Tweet features
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Ensure 3D: (B, 1, D)
        if v_img.dim() == 2:
            v_img = v_img.unsqueeze(1)
        if v_txt.dim() == 2:
            v_txt = v_txt.unsqueeze(1)

        # SAR → Tweet cross-attention
        # SAR asks: "which tweet feature dimensions confirm what I see?"
        h_img, img_attn_w = self.img_to_txt_attn(
            query=v_img, key=v_txt, value=v_txt
        )
        v_img = self.norm_img1(v_img + self.dropout(h_img))
        v_img = self.norm_img2(v_img + self.dropout(self.ffn_img(v_img)))

        # Tweet → SAR cross-attention
        # Tweet asks: "which SAR spatial features confirm what I describe?"
        h_txt, txt_attn_w = self.txt_to_img_attn(
            query=v_txt, key=v_img, value=v_img
        )
        v_txt = self.norm_txt1(v_txt + self.dropout(h_txt))
        v_txt = self.norm_txt2(v_txt + self.dropout(self.ffn_txt(v_txt)))

        return v_img.squeeze(1), v_txt.squeeze(1)  # (B, D), (B, D)


class CrossModalFusion(nn.Module):
    """
    Multi-layer cross-modal fusion with contrastive alignment.

    Args:
        d_model:   feature dimension (must match encoder output_dim)
        n_heads:   number of attention heads
        n_layers:  number of cross-attention blocks (2 recommended)
        dropout:   dropout rate
    """

    def __init__(
        self,
        d_model:  int = 512,
        n_heads:  int = 8,
        n_layers: int = 2,
        dropout:  float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossModalAttentionBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Fusion projection: concat(h_img, h_txt) → d_model
        self.fusion_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

        # Contrastive temperature (learnable)
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(
        self,
        v_img: torch.Tensor,  # (B, D)
        v_txt: torch.Tensor,  # (B, D)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            fused:  (B, D) — fused representation for classification
            h_img:  (B, D) — updated SAR features (for contrastive loss)
            h_txt:  (B, D) — updated tweet features (for contrastive loss)
        """
        h_img, h_txt = v_img, v_txt
        for layer in self.layers:
            h_img, h_txt = layer(h_img, h_txt)

        # Concatenate and project
        fused = self.fusion_proj(torch.cat([h_img, h_txt], dim=-1))  # (B, D)
        return F.normalize(fused, dim=-1), F.normalize(h_img, dim=-1), F.normalize(h_txt, dim=-1)

    def contrastive_loss(
        self,
        h_img: torch.Tensor,   # (B, D) — paired SAR features
        h_txt: torch.Tensor,   # (B, D) — paired tweet features
    ) -> torch.Tensor:
        """
        Symmetric InfoNCE contrastive loss.
        Pushes paired (SAR, tweet) closer, unpaired further.
        Based on: Radford et al. CLIP (2021).
        """
        B = h_img.shape[0]
        logits = torch.matmul(h_img, h_txt.T) / self.temperature.exp().clamp(min=0.01)
        labels = torch.arange(B, device=h_img.device)
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        return (loss_i + loss_t) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Disaster Classifier Heads
# ─────────────────────────────────────────────────────────────────────────────

class DisasterClassifier(nn.Module):
    """
    Multi-task classification heads on top of fused representation.
    Tasks:
        1. Binary:      disaster / no-disaster
        2. Type:        flood / earthquake / wildfire / cyclone / landslide / none
        3. Severity:    low / medium / high
    """

    TYPE_LABELS     = ["flood", "earthquake", "wildfire", "cyclone", "landslide", "none"]
    SEVERITY_LABELS = ["low", "medium", "high"]

    def __init__(self, d_model: int = 512, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()

        shared = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.shared = shared

        self.head_binary   = nn.Linear(hidden_dim, 2)
        self.head_type     = nn.Linear(hidden_dim, len(self.TYPE_LABELS))
        self.head_severity = nn.Linear(hidden_dim, len(self.SEVERITY_LABELS))

    def forward(self, fused: torch.Tensor) -> dict:
        """
        Args:
            fused: (B, d_model)
        Returns dict with logits:
            'binary':   (B, 2)
            'type':     (B, 6)
            'severity': (B, 3)
        """
        h = self.shared(fused)
        return {
            "binary":   self.head_binary(h),
            "type":     self.head_type(h),
            "severity": self.head_severity(h),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Full Fusion Model
# ─────────────────────────────────────────────────────────────────────────────

class DisasterFusionModel(nn.Module):
    """
    End-to-end disaster detection model.
    Combines: SAR encoder + Text encoder + TAM + Cross-attention + Classifier.

    Usage:
        model = DisasterFusionModel(sar_encoder, text_encoder, config)
        out   = model(sar_image, input_ids, attention_mask, time_offsets, credibility)
    """

    def __init__(self, sar_encoder, text_encoder, config: dict):
        super().__init__()
        self.sar_encoder  = sar_encoder
        self.text_encoder = text_encoder

        d_model = config.get("d_model", 512)

        self.tam = TemporalAlignmentModule(
            d_model=d_model,
            n_heads=config.get("n_heads_tam", 4),
            dropout=config.get("dropout", 0.1),
        )

        self.fusion = CrossModalFusion(
            d_model=d_model,
            n_heads=config.get("n_heads", 8),
            n_layers=config.get("n_layers", 2),
            dropout=config.get("dropout", 0.1),
        )

        self.classifier = DisasterClassifier(
            d_model=d_model,
            hidden_dim=config.get("hidden_dim", 256),
            dropout=config.get("dropout", 0.3),
        )

    def forward(
        self,
        sar_image:      torch.Tensor,              # (B, 2, 256, 256)
        tweet_embeds:   torch.Tensor,              # (B, T, D) pre-encoded
        time_offsets:   torch.Tensor,              # (B, T)
        credibility:    torch.Tensor,              # (B, T)
        tweet_mask:     Optional[torch.Tensor] = None,  # (B, T)
        # Transformer-mode tweet input (if encoding on-the-fly)
        input_ids:      Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:

        # ── Encode SAR ──────────────────────────────────────────────────────
        v_img = self.sar_encoder(sar_image)  # (B, D)

        # ── Encode Tweets (if raw input_ids provided) ────────────────────────
        if input_ids is not None:
            B, T, L = input_ids.shape
            # Flatten batch×window, encode, reshape
            flat_ids   = input_ids.view(B * T, L)
            flat_mask  = attention_mask.view(B * T, L) if attention_mask is not None else None
            flat_emb   = self.text_encoder(flat_ids, flat_mask)  # (B*T, D)
            tweet_embeds = flat_emb.view(B, T, -1)               # (B, T, D)

        # ── Temporal Alignment ───────────────────────────────────────────────
        v_txt = self.tam(tweet_embeds, time_offsets, credibility, tweet_mask)  # (B, D)

        # ── Cross-Modal Fusion ───────────────────────────────────────────────
        fused, h_img, h_txt = self.fusion(v_img, v_txt)  # (B, D)

        # ── Classify ─────────────────────────────────────────────────────────
        logits = self.classifier(fused)

        return {
            **logits,
            "fused":  fused,
            "h_img":  h_img,
            "h_txt":  h_txt,
        }


if __name__ == "__main__":
    B, T, D = 4, 10, 512

    # Test TAM
    tam = TemporalAlignmentModule(d_model=D, n_heads=4)
    tweet_emb = F.normalize(torch.randn(B, T, D), dim=-1)
    offsets   = torch.randint(0, 60, (B, T))
    cred      = torch.rand(B, T)
    mask      = torch.ones(B, T, dtype=torch.bool)
    mask[0, 7:] = False
    aligned = tam(tweet_emb, offsets, cred, mask)
    print(f"TAM output: {aligned.shape}")

    # Test Cross-Attention Fusion
    fusion = CrossModalFusion(d_model=D, n_heads=8, n_layers=2)
    v_img  = F.normalize(torch.randn(B, D), dim=-1)
    fused, h_img, h_txt = fusion(v_img, aligned)
    print(f"Fusion output: {fused.shape}")

    # Test Contrastive Loss
    cl = fusion.contrastive_loss(h_img, h_txt)
    print(f"Contrastive loss: {cl.item():.4f}")

    # Test Classifier
    clf = DisasterClassifier(d_model=D)
    out = clf(fused)
    print(f"Binary logits: {out['binary'].shape}")
    print(f"Type logits:   {out['type'].shape}")
    print(f"Severity logits: {out['severity'].shape}")

    print("All fusion module tests passed.")
