"""
Model builder for the ReID network.

Wraps a pretrained CLIP image encoder into a lightweight PyTorch module
and exposes a factory function ``build_model`` that the inference engine calls.
"""

import torch
import torch.nn as nn
from .clip_model import build_CLIP_from_openai_pretrained


class ReID(nn.Module):
    """
    ReID network wrapping a pretrained CLIP visual encoder.

    Attributes:
        base_model : nn.Module – the underlying CLIP image encoder.
        logit_scale: float      – learned temperature scaling parameter.
        embed_dim  : int        – dimensionality of the output feature vector.
    """

    def __init__(self, args):
        super().__init__()
        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(
            args.pretrain_choice, args.img_size, args.stride_size
        )
        self.logit_scale = torch.ones([]) * (1 / args.temperature)
        self.embed_dim = base_cfg['embed_dim']

    def encode_image(self, image):
        """Extract feature embedding for the given image tensor(s)."""
        return self.base_model.encode_image(image)


def build_model(args):
    """Factory function: instantiate the ReID model from parsed CLI / YAML args."""
    return ReID(args)
