"""
OpenAI CLIP model implementation adapted for person Re-ID.

This module provides:
  - Vision backbones: ModifiedResNet and VisionTransformer
  - Full CLIP wrapper with image + text encoders
  - Pretrained weight loading from OpenAI release URLs
  - Position-embedding interpolation for arbitrary input resolutions

Reference: https://github.com/openai/CLIP
"""

from collections import OrderedDict
import logging
import math
import os
from typing import List, Tuple, Union
import hashlib
import urllib
from tqdm import tqdm
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger("ReID")

# ---------------------------------------------------------------------------
# OpenAI CLIP pretrained model URLs (keyed by model name)
# ---------------------------------------------------------------------------
_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
}


def available_models() -> List[str]:
    """Return the list of supported CLIP model names."""
    return list(_MODELS.keys())


def _download(url: str, root: str):
    """
    Download a model checkpoint from ``url`` into ``root``.

    SHA-256 verification is performed against the checksum embedded in the URL.
    """
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    # Verify existing file by SHA-256 checksum.
    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} checksum mismatch; re-downloading")

    # Download with progress bar.
    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length")),
            ncols=80, unit='iB', unit_scale=True, unit_divisor=1024
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break
                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Downloaded model SHA256 checksum does not match")

    return download_target


# ---------------------------------------------------------------------------
# ResNet backbone components
# ---------------------------------------------------------------------------

class Bottleneck(nn.Module):
    """Standard ResNet Bottleneck block (expansion = 4)."""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # All conv layers default to stride=1; avgpool handles stride > 1.
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # Downsampling: avgpool first, then 1x1 conv with stride=1.
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    """
    Multi-head attention pooling over spatial features.

    Used as the final pooling layer in ModifiedResNet instead of global avgpool.
    """
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn((spacial_dim[0] * spacial_dim[1]) + 1, embed_dim) / embed_dim ** 0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        # NCHW -> (HW)NC
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)
        # Prepend global average token -> (HW+1)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        # Add learned positional embeddings.
        x = x + self.positional_embedding[:, None, :].to(x.dtype)
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None, bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x[0]  # Return the global token only.


class ModifiedResNet(nn.Module):
    """
    ResNet variant used by CLIP.

    Differences from torchvision ResNet:
      - 3-layer stem convolutions (instead of 1) with avgpool
      - Anti-aliased strided convolutions (avgpool before conv)
      - Attention-based final pooling (AttentionPool2d)
    """
    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # Residual stages
        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # ResNet feature dimension = 2048
        spacial_dim = (
            input_resolution[0] // 32,
            input_resolution[1] // 32,
        )
        self.attnpool = AttentionPool2d(spacial_dim, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)
        return x


# ---------------------------------------------------------------------------
# Vision Transformer (ViT) backbone components
# ---------------------------------------------------------------------------

class LayerNorm(nn.LayerNorm):
    """LayerNorm that casts input to fp32 before normalizing (for mixed-precision training)."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    """Fast GELU approximation: x * sigmoid(1.702 * x)."""
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """Standard Transformer encoder block with pre-norm and causal attention mask support."""
    def __init__(self, d_model: int, n_head: int, attn_mask=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        attn_mask_ = self.attn_mask
        if self.attn_mask is not None and hasattr(self.attn_mask, '__call__'):
            attn_mask_ = self.attn_mask(x.size(0))  # LND layout

        attn_mask_ = attn_mask_.to(dtype=x.dtype, device=x.device) if attn_mask_ is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """Stack of residual attention blocks."""
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor):
        for layer in self.resblocks:
            x = layer(x)
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer adapted for CLIP image encoding.

    Supports sliding-window patch extraction via ``stride_size`` (for ReID tasks).
    """
    def __init__(self, input_resolution: Tuple[int, int], patch_size: int, stride_size: int,
                 width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.num_x = (input_resolution[1] - patch_size) // stride_size + 1
        self.num_y = (input_resolution[0] - patch_size) // stride_size + 1
        num_patches = self.num_x * self.num_y

        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=width,
            kernel_size=patch_size, stride=stride_size, bias=False
        )

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(num_patches + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        # Patch embedding
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        # Prepend CLS token
        x = torch.cat([
            self.class_embedding.to(x.dtype)
            + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        # Transformer expects (L, N, C) layout
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)

        x = self.ln_post(x)

        if self.proj is not None:
            x = x @ self.proj

        return x[:, 0, :].float()  # Return CLS-token embedding.


def resize_text_pos_embed(v, context_length):
    """Placeholder for text positional embedding resizing (not used in ReID)."""
    pass


# ---------------------------------------------------------------------------
# Full CLIP wrapper
# ---------------------------------------------------------------------------

class CLIP(nn.Module):
    """
    Contrastive Language-Image Pre-training (CLIP) model.

    The text encoder is kept for compatibility; only ``encode_image`` is used
    in this ReID codebase.
    """
    def __init__(self,
                 embed_dim: int,
                 # Vision hyper-parameters
                 image_resolution: Union[int, Tuple[int, int]],
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 stride_size: int,
                 # Text hyper-parameters
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        # Choose between ResNet and Vision Transformer based on vision_layers type.
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                stride_size=stride_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        # Text transformer (unused in pure ReID inference but kept for weight loading).
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

        self.initialize_parameters()

    def initialize_parameters(self):
        """Initialize all learnable parameters with small Gaussian noise."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            # Zero-init the final BN gamma in each Bottleneck for stable training.
            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        """Build a causal (lower-triangular) attention mask for the text transformer."""
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # Zero out the lower diagonal (causal masking)
        return mask

    @property
    def dtype(self):
        """Return the data type of the vision backbone's first conv weight."""
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        """Encode image tensor(s) into fixed-size feature vector(s)."""
        return self.visual(image.type(self.dtype))

    def forward(self, tar_image, can_image):
        """Dual-image forward used during training (target + candidate)."""
        tar_feat = self.encode_image(tar_image)
        can_feat = self.encode_image(can_image)
        return tar_feat, can_feat

    def load_param(self, state_dict):
        """
        Load parameters from a pretrained checkpoint with shape mismatch handling.

        Supports position-embedding interpolation when input resolution differs
        from the pretrained model.
        """
        param_dict = {k: v for k, v in state_dict.items() if k in self.state_dict()}

        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']

        for k, v in param_dict.items():
            # Interpolate vision positional embeddings if shape mismatch.
            if k == 'visual.positional_embedding' and v.shape != self.visual.positional_embedding.shape:
                v = resize_pos_embed(v, self.visual.positional_embedding, self.visual.num_y, self.visual.num_x)
            elif k == 'positional_embedding' and v.shape != self.positional_embedding.shape:
                v = resize_text_pos_embed(v, self.context_length)
            try:
                self.state_dict()[k].copy_(v)
            except:
                print(f'===========================ERROR during copy {k}, {v.shape}=========================')
                print(f'shape mismatch in k:{k}: param_dict{v.shape} vs self.state_dict(){self.state_dict()[k].shape}')


def resize_pos_embed(posemb, posemb_new, height, width):
    """
    Interpolate 2D positional embeddings to a new spatial resolution.

    Args:
        posemb    : old positional embedding tensor (1 + H*W, C)
        posemb_new: new positional embedding parameter to match
        height    : target grid height
        width     : target grid width

    Returns:
        Resized positional embedding of shape (1 + height*width, C).
    """
    posemb = posemb.unsqueeze(0)
    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]

    gs_old = int(math.sqrt(len(posemb_grid)))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(height, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, height * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb.squeeze(0)


def convert_weights(model: nn.Module):
    """Convert applicable model parameters (Conv, Linear, MHA) to fp16."""
    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                         "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj", "mcq_proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_CLIP_from_openai_pretrained(
    name: str,
    image_size: Union[int, Tuple[int, int]],
    stride_size: int,
    jit: bool = False,
    download_root: str = None
):
    """
    Build a CLIP model from OpenAI pretrained weights.

    Automatically infers architecture hyper-parameters from the checkpoint and
    swaps in the desired input resolution / stride.

    Args:
        name         : model name (e.g. "ViT-B/16") or local checkpoint path.
        image_size   : desired input resolution (H, W) for ReID.
        stride_size  : patch stride for VisionTransformer (overrides pretrained).
        jit          : whether the checkpoint is a TorchScript archive.
        download_root: directory to cache downloaded weights.

    Returns:
        (model, model_cfg) where model_cfg is a dict of inferred hyper-parameters.
    """
    # Resolve model source
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    # Load checkpoint
    try:
        model = torch.jit.load(model_path, map_location="cpu")
        state_dict = None
    except RuntimeError:
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

    state_dict = state_dict or model.state_dict()

    # Infer architecture from checkpoint keys
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model_cfg = {
        'embed_dim': embed_dim,
        'image_resolution': image_resolution,
        'vision_layers': vision_layers,
        'vision_width': vision_width,
        'vision_patch_size': vision_patch_size,
        'context_length': context_length,
        'vocab_size': vocab_size,
        'transformer_width': transformer_width,
        'transformer_heads': transformer_heads,
        'transformer_layers': transformer_layers
    }

    # Override resolution / stride with user-provided ReID settings
    model_cfg['image_resolution'] = image_size
    model_cfg['stride_size'] = stride_size
    logger.info(f"Loading pretrained {name} CLIP model with config: {model_cfg}")
    model = CLIP(**model_cfg)

    # Uncomment to enable fp16 inference
    # convert_weights(model)

    model.load_param(state_dict)
    return model, model_cfg
