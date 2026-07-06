"""
models/encoders/text_encoder.py
────────────────────────────────
Tweet text encoder using BERTweet (pre-trained on 850M tweets).
Outputs 512-dim sentence embeddings pooled from [CLS] token.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict
from loguru import logger

try:
    from transformers import (
        AutoTokenizer,
        AutoModel,
        RobertaModel,
        RobertaTokenizer,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not installed. Using simple text encoder.")


class TextEncoder(nn.Module):
    """
    Encodes a batch of tweet texts → (B, output_dim) feature vectors.

    Supported models:
        'vinai/bertweet-base'  — pretrained on 850M English tweets (recommended)
        'roberta-base'         — general-purpose, strong baseline
        'simple_bow'           — fallback bag-of-words for testing

    Args:
        model_name:   HuggingFace model name or 'simple_bow'
        output_dim:   feature dimension (512 recommended, same as SAREncoder)
        max_length:   max token length (128 for tweets)
        freeze_layers: number of transformer layers to freeze
        pooling:      'cls' | 'mean' — how to pool token embeddings
    """

    def __init__(
        self,
        model_name:    str = "vinai/bertweet-base",
        output_dim:    int = 512,
        max_length:    int = 128,
        freeze_layers: int = 4,
        pooling:       str = "cls",
    ):
        super().__init__()
        self.model_name  = model_name
        self.output_dim  = output_dim
        self.max_length  = max_length
        self.pooling     = pooling

        if TRANSFORMERS_AVAILABLE and model_name != "simple_bow":
            self._build_transformer(model_name, freeze_layers)
        else:
            self._build_bow_encoder()

    def _build_transformer(self, model_name: str, freeze_layers: int):
        logger.info(f"Loading text encoder: {model_name}")
        try:
            self.tokenizer  = AutoTokenizer.from_pretrained(
                model_name,
                normalization=True,  # BERTweet-specific
                use_fast=True,
            )
            self.transformer= AutoModel.from_pretrained(model_name)
            feat_dim        = self.transformer.config.hidden_size

            # Freeze early layers
            if freeze_layers > 0:
                self._freeze_layers(freeze_layers)

            # Projection: bert_dim → output_dim
            self.projector = nn.Sequential(
                nn.Linear(feat_dim, output_dim * 2),
                nn.LayerNorm(output_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(output_dim * 2, output_dim),
                nn.LayerNorm(output_dim),
            )
            self.use_transformer = True
            logger.info(
                f"TextEncoder: {model_name} | hidden={feat_dim} "
                f"→ {output_dim} | frozen={freeze_layers} layers"
            )
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}. Using BOW fallback.")
            self._build_bow_encoder()

    def _freeze_layers(self, n: int):
        """Freeze first n encoder layers."""
        params = list(self.transformer.encoder.layer[:n].parameters())
        for p in params:
            p.requires_grad = False
        # Always keep embedding layer trainable for domain adaptation
        for p in self.transformer.embeddings.parameters():
            p.requires_grad = True

    def _build_bow_encoder(self):
        """Simple bag-of-words fallback encoder for testing without GPU/internet."""
        logger.info("TextEncoder: Using simple BOW fallback encoder")
        vocab_size = 30000
        embed_dim  = 128
        self.embedding   = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")
        self.projector   = nn.Sequential(
            nn.Linear(embed_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
        )
        self.use_transformer = False

        # Simple character-based tokeniser for testing
        self.vocab = {}
        self.vocab_size = vocab_size

    def _tokenize_bow(self, texts: List[str], device: torch.device):
        """Tokenise texts for BOW fallback using simple word hash."""
        max_words = 30
        indices = []
        for text in texts:
            words = text.lower().split()[:max_words]
            ids   = [hash(w) % self.vocab_size for w in words] if words else [0]
            indices.append(torch.tensor(ids, dtype=torch.long))
        offsets = torch.tensor([0] + [len(x) for x in indices[:-1]]).cumsum(0)
        flat    = torch.cat(indices)
        return flat.to(device), offsets.to(device)

    def tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenise a list of tweet strings."""
        if not self.use_transformer:
            return {"texts": texts}
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        texts: Optional[List[str]] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args (transformer mode):
            input_ids:      (B, seq_len)
            attention_mask: (B, seq_len)
        Args (BOW mode):
            texts: list of strings
        Returns:
            embeddings: (B, output_dim) L2-normalised
        """
        if not self.use_transformer:
            device = next(self.parameters()).device
            if texts is None:
                raise ValueError("texts required for BOW encoder")
            flat, offsets = self._tokenize_bow(texts, device)
            feats = self.embedding(flat, offsets)           # (B, embed_dim)
            projected = self.projector(feats)
            return F.normalize(projected, dim=-1)

        # Transformer forward
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None,
        )

        if self.pooling == "cls":
            # [CLS] token representation
            feats = outputs.last_hidden_state[:, 0, :]     # (B, hidden_size)
        else:
            # Mean pooling over non-padding tokens
            mask    = attention_mask.unsqueeze(-1).float()
            feats   = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        projected = self.projector(feats)                   # (B, output_dim)
        return F.normalize(projected, dim=-1)


class TweetWindowEncoder(nn.Module):
    """
    Encodes a window of tweets (variable length) into a single fixed-size vector.
    Used in the Temporal Alignment Module to represent all tweets
    in a SAR-aligned time window.

    Approach: encode each tweet → attend over credibility → weighted mean pool.
    """

    def __init__(self, text_encoder: TextEncoder, output_dim: int = 512):
        super().__init__()
        self.encoder = text_encoder
        self.cred_attention = nn.Sequential(
            nn.Linear(output_dim + 1, 128),   # +1 for credibility score
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        tweet_embeddings: torch.Tensor,         # (B, T, output_dim)
        credibility_scores: torch.Tensor,       # (B, T)
        tweet_mask: Optional[torch.Tensor] = None,  # (B, T) True=valid
    ) -> torch.Tensor:
        """
        Args:
            tweet_embeddings:  (B, T, D) — pre-encoded tweet vectors
            credibility_scores:(B, T)    — credibility scores in [0,1]
            tweet_mask:        (B, T)    — mask for padding
        Returns:
            window_embedding: (B, D)
        """
        B, T, D = tweet_embeddings.shape
        cred     = credibility_scores.unsqueeze(-1)              # (B, T, 1)
        combined = torch.cat([tweet_embeddings, cred], dim=-1)   # (B, T, D+1)

        # Attention weights based on content + credibility
        attn_scores = self.cred_attention(combined).squeeze(-1)  # (B, T)

        if tweet_mask is not None:
            attn_scores = attn_scores.masked_fill(~tweet_mask, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)        # (B, T)
        pooled = (tweet_embeddings * attn_weights.unsqueeze(-1)).sum(1)  # (B, D)
        return F.normalize(pooled, dim=-1)


if __name__ == "__main__":
    # Test BOW fallback (no dependencies needed)
    logger.info("Testing TextEncoder with BOW fallback...")
    enc = TextEncoder(model_name="simple_bow", output_dim=512)
    texts = [
        "Flooding in Mumbai roads completely underwater #flood",
        "Earthquake felt across Istanbul buildings shaking #earthquake",
        "Normal sunny day in Delhi nothing happening",
    ]
    enc_out = enc(texts=texts)
    print(f"BOW TextEncoder output: {enc_out.shape}")
    assert enc_out.shape == (3, 512)

    # Test TweetWindowEncoder
    window_enc = TweetWindowEncoder(enc, output_dim=512)
    embeds = torch.randn(2, 5, 512)                   # 2 windows, 5 tweets each
    embeds = F.normalize(embeds, dim=-1)
    creds  = torch.rand(2, 5)
    mask   = torch.ones(2, 5, dtype=torch.bool)
    mask[0, 3:] = False   # window 0 only has 3 valid tweets
    out    = window_enc(embeds, creds, mask)
    print(f"TweetWindowEncoder output: {out.shape}")
    assert out.shape == (2, 512)

    print("TextEncoder tests passed.")
