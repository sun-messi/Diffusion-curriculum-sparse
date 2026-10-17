"""Lightweight curriculum components; no training or dataset imports."""
import math
import torch
from torch import nn

class EmbeddingSparsityMask(nn.Module):
    """
    Apply progressive sparsity mask to embeddings.

    train_c.py attaches this mask only after PatchEmbed.
    The decoder and transformer blocks are not independently masked.

    This creates hierarchical feature learning:
    - Early stages (high noise, high sparsity): Only first K dims active
    - Later stages (low noise, low sparsity): More dims activated
    - Inference: All dims active (no mask)
    """
    def __init__(self, embed_dim):
        super().__init__()
        if not isinstance(embed_dim, int) or isinstance(embed_dim, bool) or embed_dim < 1:
            raise ValueError('embed_dim must be a positive integer')
        self.embed_dim = embed_dim
        self.current_sparsity = 0.0
        self.is_inference = False

    def set_sparsity(self, sparsity):
        """Update current sparsity level."""
        sparsity = float(sparsity)
        if not math.isfinite(sparsity) or not 0.0 <= sparsity <= 1.0:
            raise ValueError('sparsity must be finite and in [0, 1]')
        self.current_sparsity = sparsity

    def set_inference_mode(self, is_inference):
        """Set inference mode (disables masking)."""
        self.is_inference = is_inference

    def forward(self, x):
        """
        Apply sparsity mask to embeddings.

        Args:
            x: [..., embed_dim] embeddings

        Returns:
            Masked embeddings with same shape
        """
        if x.shape[-1] != self.embed_dim:
            raise ValueError('Input final dimension must match embed_dim')
        # No masking during inference or when sparsity=0
        if self.is_inference or self.current_sparsity == 0.0:
            return x

        # Calculate active dimensions
        active_dims = int(self.embed_dim * (1.0 - self.current_sparsity))
        active_dims = max(1, min(active_dims, self.embed_dim))

        # Create and apply mask
        mask = torch.zeros(self.embed_dim, device=x.device, dtype=x.dtype)
        mask[:active_dims] = 1.0

        return x * mask  # Zero out inactive dimensions

def validate_time_range(t_min, t_max):
    if not (math.isfinite(t_min) and math.isfinite(t_max) and 0 <= t_min < t_max <= 1):
        raise ValueError('Expected finite 0 <= t_min < t_max <= 1')
