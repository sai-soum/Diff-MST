# mst/utils/variable_length.py  (create this file anywhere on your PYTHONPATH)
import torch
import torch.nn.functional as F
from typing import Literal, Optional

class AttentionPool(torch.nn.Module):
    """Learned attention-pooling over a sequence of chunk embeddings."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.attn = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, embed_dim // 2),
            torch.nn.Tanh(),
            torch.nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # x: (B, N, D)
        e = self.attn(x).squeeze(-1)          # (B, N)
        if mask is not None:
            e = e.masked_fill(mask, -1e9)
        α = torch.softmax(e, dim=-1).unsqueeze(-1)  # (B, N, 1)
        return (α * x).sum(dim=1)             # (B, D)


class VariableLengthEncoder(torch.nn.Module):
    """
    Wraps an existing CNN spectrogram encoder so it can accept
    arbitrary-length waveforms by chunking + pooling.
    """
    def __init__(
        self,
        base_encoder: torch.nn.Module,
        *,
        sr: int = 44_100,
        chunk_seconds: float = 5.0,
        hop_seconds: Optional[float] = None,
        pool: Literal["mean", "max", "attention"] = "mean",
    ):
        super().__init__()
        self.base_encoder = base_encoder  # pretrained weights stay intact
        self.sr = sr
        self.chunk_len = int(chunk_seconds * sr)
        self.hop_len = int((hop_seconds or chunk_seconds) * sr)

        if pool == "mean":
            self.pool = lambda x, m=None: x.mean(dim=1)
        elif pool == "max":
            self.pool = lambda x, m=None: x.max(dim=1).values
        elif pool == "attention":
            self.pool = AttentionPool(base_encoder.embed_dim)
        else:
            raise ValueError(f"Unknown pool mode: {pool}")

    @torch.no_grad()               # we don’t change encoder weights
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T) waveform
        returns: (B, embed_dim) pooled embedding
        """
        B, C, T = x.shape
        if T <= self.chunk_len:
            return self.base_encoder(x)

        # Build indices for chunking
        starts = torch.arange(0, T - self.chunk_len + 1, self.hop_len, device=x.device)
        # Pad last chunk if needed so every sample is covered
        if starts[-1] + self.chunk_len < T:
            starts = torch.cat([starts, starts.new_tensor([T - self.chunk_len])])

        # Extract chunks -> (total_chunks, C, chunk_len)
        chunks = torch.stack([x[..., s : s + self.chunk_len] for s in starts], dim=0)
        chunks = chunks.reshape(-1, C, self.chunk_len)   # flatten batch
        embeds = self.base_encoder(chunks)               # (total_chunks, D)

        # reshape back to (B, N, D)
        num_chunks = starts.shape[0]
        embeds = embeds.view(B, num_chunks, -1)
        # Optional mask (only matters if hop_len doesn’t evenly tile)
        pooled = self.pool(embeds)
        return pooled
