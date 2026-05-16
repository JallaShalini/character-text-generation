"""
Mini Transformer model for character-level text generation.
Compact CPU-friendly architecture with custom self-attention and encoder blocks.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding added to token embeddings."""

    def __init__(self, embedding_dim, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float)
            * (-math.log(10000.0) / embedding_dim)
        )
        pe = torch.zeros(max_len, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    """Custom multi-head self-attention for encoder blocks."""

    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.query_proj = nn.Linear(embedding_dim, embedding_dim)
        self.key_proj = nn.Linear(embedding_dim, embedding_dim)
        self.value_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def _shape_projection(self, tensor, batch_size, seq_length):
        return tensor.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, attn_mask=None):
        batch_size, seq_length, _ = x.shape

        queries = self._shape_projection(self.query_proj(x), batch_size, seq_length)
        keys = self._shape_projection(self.key_proj(x), batch_size, seq_length)
        values = self._shape_projection(self.value_proj(x), batch_size, seq_length)

        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        attended = torch.matmul(attention_weights, values)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embedding_dim)
        return self.out_proj(attended)


class FeedForward(nn.Module):
    """Position-wise feed-forward block."""

    def __init__(self, embedding_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerEncoderLayer(nn.Module):
    """Single Transformer encoder layer with residuals and layer normalization."""

    def __init__(self, embedding_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(embedding_dim, num_heads, dropout)
        self.feed_forward = FeedForward(embedding_dim, ffn_dim, dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        attn_output = self.self_attn(x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout(attn_output))

        ffn_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x


class TransformerModel(nn.Module):
    """Compact mini Transformer for character-level next-token prediction."""

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        num_heads=4,
        ffn_dim=512,
        num_layers=2,
        dropout=0.1,
        max_len=5000,
    ):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_encoding = PositionalEncoding(embedding_dim, max_len=max_len, dropout=dropout)
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(embedding_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, vocab_size)

        print("TransformerModel initialized:")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  Heads: {num_heads}")
        print(f"  FFN dim: {ffn_dim}")
        print(f"  Layers: {num_layers}")
        print(f"  Dropout: {dropout}")
        print(f"  Total parameters: {self.count_parameters():,}")

    def count_parameters(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _generate_causal_mask(self, seq_length, device):
        return torch.tril(torch.ones(seq_length, seq_length, device=device, dtype=torch.bool))

    def forward(self, x, attn_mask=None):
        """Return logits with shape (batch_size, seq_length, vocab_size)."""
        batch_size, seq_length = x.shape

        x = self.token_embedding(x)
        x = self.position_encoding(x)

        if attn_mask is None:
            attn_mask = self._generate_causal_mask(seq_length, x.device)

        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)

        x = self.final_norm(x)
        logits = self.output_projection(x)
        return logits, None

    def generate(self, seed_tokens, max_length=100, temperature=1.0):
        """Autoregressive sampling helper for later phases."""
        self.eval()
        generated = seed_tokens.clone()

        with torch.no_grad():
            for _ in range(max_length):
                logits, _ = self.forward(generated)
                next_logits = logits[:, -1, :] / temperature
                probabilities = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)

        return generated


def create_transformer_model(
    vocab_size,
    embedding_dim=128,
    num_heads=4,
    ffn_dim=512,
    num_layers=2,
    dropout=0.1,
    max_len=5000,
):
    """Factory helper for the mini Transformer."""
    return TransformerModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        num_layers=num_layers,
        dropout=dropout,
        max_len=max_len,
    )