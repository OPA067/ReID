"""Contrastive loss objectives for person re-identification."""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Objective"]


class Objective(nn.Module):
    """
    Contrastive loss module for person re-identification.

    Computes an InfoNCE (noise contrastive estimation) loss that pulls
    feature embeddings of the same person together while pushing apart
    embeddings of different persons within a batch.

    Args:
        logit_scale (float): Temperature scaling factor applied to the
            similarity scores before softmax.  A lower temperature makes
            the distribution sharper (more discriminative).
        loss_type (str): Contrastive loss variant (default: 'InfoNCE').
        embed_dim (int): Dimensionality of the feature embedding vectors.
    """

    def __init__(self, logit_scale=50, loss_type='InfoNCE', embed_dim=512):
        super().__init__()
        self.logit_scale = logit_scale
        self.loss_type = loss_type
        self.embed_dim = embed_dim

    def _norm(self, x):
        """
        Apply L2 normalization along the feature dimension.

        Args:
            x (Tensor): Input feature tensor, shape [B, D].

        Returns:
            Tensor: L2-normalized features, shape [B, D].
        """
        return x / x.norm(dim=-1, keepdim=True)

    def _compute_InfoNCE_per(self, scores):
        """
        Compute symmetric InfoNCE loss per sample.

        Given a similarity matrix scores of shape [B, B], where the
        diagonal entries correspond to positive pairs (same person),
        this function minimizes:
            -log(softmax(scores, dim=1)[diag])  (image->target branch)
            -log(softmax(scores.T, dim=1)[diag]) (target->image branch)
        and returns the averaged loss per row.

        Args:
            scores (Tensor): Cosine similarity matrix, shape [B, B].

        Returns:
            Tensor: Per-sample InfoNCE loss, shape [B].
        """
        logits = self.logit_scale * scores
        logits_t = logits.t()

        p1 = F.softmax(logits, dim=1)
        p2 = F.softmax(logits_t, dim=1)

        loss = (-p1.diag().log() - p2.diag().log()) / 2
        return loss

    def _compute_loss(self, tar_feat, can_feat):
        """
        Compute the total contrastive loss between target and candidate features.

        Normalizes both feature sets, computes the similarity matrix, and
        applies InfoNCE by treating the diagonal as positive pairs.

        Args:
            tar_feat (Tensor): Target person features, shape [B, D].
            can_feat (Tensor): Candidate person features, shape [B, D].

        Returns:
            Tensor: Scalar total loss across the batch.
        """
        tar_feat = self._norm(tar_feat)
        can_feat = self._norm(can_feat)
        scores = tar_feat @ can_feat.t()
        loss = self._compute_InfoNCE_per(scores).sum()
        return loss

    def forward(self, tar_feat, can_feat):
        """
        Alias for _compute_loss.

        Args:
            tar_feat (Tensor): Target features.
            can_feat (Tensor): Candidate features.

        Returns:
            Tensor: Scalar loss.
        """
        return self._compute_loss(tar_feat, can_feat)