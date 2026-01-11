"""
Feature Extractor for U-ViT models.

Uses PyTorch forward hooks to extract intermediate layer activations.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict


class FeatureExtractor:
    """
    Context manager for extracting intermediate features from U-ViT models.

    Usage:
        with FeatureExtractor(model) as extractor:
            extractor.register_layer("bottleneck", "mid_block")
            features = extractor.extract_features(images, timesteps)
            bottleneck_feat = features["bottleneck"]
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.features: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._layer_mapping: Dict[str, str] = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()
        return False

    def _get_module(self, module_path: str) -> nn.Module:
        """
        Get a module by its dot-separated path.

        Args:
            module_path: e.g., "mid_block", "in_blocks.5", "out_blocks.0"
        """
        parts = module_path.split('.')
        module = self.model

        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)

        return module

    def register_layer(self, layer_name: str, module_path: str):
        """
        Register a layer for feature extraction.

        Args:
            layer_name: Name to use for this layer in the features dict
            module_path: Path to the module, e.g., "mid_block", "in_blocks.5"
        """
        self._layer_mapping[layer_name] = module_path
        module = self._get_module(module_path)

        def hook_fn(name):
            def fn(module, input, output):
                # Detach and move to CPU to save GPU memory
                if isinstance(output, torch.Tensor):
                    self.features[name] = output.detach().cpu()
                elif isinstance(output, tuple):
                    self.features[name] = output[0].detach().cpu()
            return fn

        handle = module.register_forward_hook(hook_fn(layer_name))
        self.hooks.append(handle)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
        self._layer_mapping.clear()

    def extract_features(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Run forward pass and return extracted features.

        Args:
            x: Input images [B, C, H, W]
            t: Timesteps [B]

        Returns:
            Dict mapping layer names to feature tensors
        """
        self.features.clear()

        with torch.no_grad():
            _ = self.model(x, t)

        return dict(self.features)

    def clear(self):
        """Clear stored features."""
        self.features.clear()


def get_uvit_target_layers() -> List[Tuple[str, str]]:
    """
    Get pre-defined layer extraction targets for U-ViT.

    Returns:
        List of (layer_name, module_path) tuples
    """
    return [
        ("in_block_0", "in_blocks.0"),
        ("in_block_2", "in_blocks.2"),
        ("in_block_5", "in_blocks.5"),
        ("mid_block", "mid_block"),
        ("out_block_0", "out_blocks.0"),
        ("out_block_2", "out_blocks.2"),
        ("out_block_5", "out_blocks.5"),
    ]
