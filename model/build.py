import torch
import torch.nn as nn
from .clip_model import build_CLIP_from_openai_pretrained
from .objectives import Objective


class reid(nn.Module):
    """
    Person Re-Identification (ReID) model built on top of a CLIP visual backbone.

    The model extracts image features using a pre-trained CLIP visual encoder,
    then computes a contrastive InfoNCE loss between target and candidate images.
    This encourages the model to map images of the same person to similar feature
    representations and images of different persons to dissimilar features.

    Args:
        args: Configuration object containing:
            - pretrain_choice: Name of the CLIP model to load (e.g., 'ViT-B/32').
            - img_size: Input image resolution as (height, width).
            - stride_size: Patch stride for the vision transformer.
            - temperature: Temperature hyperparameter for InfoNCE scaling.
            - loss_names: Names of loss functions to use, separated by '+'.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self._set_task()

        # ------------------------------------------------------------------
        # Build the visual backbone from OpenAI's pre-trained CLIP checkpoint.
        # Only the visual encoder is used (text encoder is omitted for
        # image-to-image ReID).
        # ------------------------------------------------------------------
        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(
            args.pretrain_choice, args.img_size, args.stride_size,
        )
        self.logit_scale = torch.ones([]) * (1 / args.temperature)
        self.loss_type = 'InfoNCE'
        self.embed_dim = base_cfg['embed_dim']

        # ------------------------------------------------------------------
        # Contrastive loss module for computing the InfoNCE loss between
        # target and candidate feature vectors.
        # ------------------------------------------------------------------
        self.loss_fn = Objective(
            logit_scale=self.logit_scale,
            loss_type=self.loss_type,
            embed_dim=self.embed_dim,
        )

    def _set_task(self):
        """
        Parse the loss_names argument into a list of task names.

        Example:
            loss_names='sdm+id+mlm' -> current_task=['sdm', 'id', 'mlm']
        """
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]

    def encode_image(self, image):
        """
        Encode a single image into a feature vector.

        Args:
            image (Tensor): Input image tensor of shape [B, 3, H, W].

        Returns:
            Tensor: L2-normalized feature embedding of shape [B, embed_dim].
        """
        feat = self.base_model.encode_image(image)
        return feat

    def forward(self, tar_img, can_img):
        """
        Forward pass: extract features for both target and candidate images,
        then compute the contrastive loss.

        Args:
            tar_img (Tensor): Target image tensor, shape [B, 3, H, W].
            can_img (Tensor): Candidate image tensor, shape [B, 3, H, W].

        Returns:
            dict: Contains 'temperature' (1 / logit_scale) and 'loss'.
        """
        ret = {'temperature': 1 / self.logit_scale}

        # ------------------------------------------------------------------
        # Extract visual features from target and candidate images.
        # Both share the same visual encoder (Siamese-like architecture).
        # ------------------------------------------------------------------
        tar_feat, can_feat = self.base_model(tar_img, can_img)
        loss = self.loss_fn(tar_feat, can_feat)

        ret.update({'loss': loss})
        return ret


def build_model(args):
    """
    Factory function to instantiate a ReID model from configuration.

    Args:
        args: Parsed configuration object.

    Returns:
        nn.Module: An instance of the reid model.
    """
    model = reid(args)
    return model
