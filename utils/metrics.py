"""
Evaluation metrics for person re-identification.

Provides common CMC (Cumulative Matching Characteristics) metrics used in
ReID literature, such as Rank-1, Rank-5, Rank-10 accuracy, mean / median rank,
and RSum (a composite metric).
"""

from prettytable import PrettyTable
import torch
import numpy as np
import logging
from tqdm import tqdm


def metrics(similarity_matrix):
    """
    Compute ranking metrics from a similarity matrix.

    For each query (row), sorts all gallery candidates by similarity in
    descending order and records the rank at which the ground-truth match
    (diagonal entry) appears.

    Args:
        similarity_matrix (numpy.ndarray): Square similarity matrix of shape
            [N, N]. The element at [i, j] represents the similarity between
            query i and gallery candidate j. The diagonal corresponds to the
            ground-truth match (same person).

    Returns:
        dict: Dictionary of ranking metrics:
            - R1:  Rank-1 accuracy (%)
            - R5:  Rank-5 accuracy (%)
            - R10: Rank-10 accuracy (%)
            - Rsum: Sum of R1, R5, and R10 (composite metric)
            - R50: Rank-50 accuracy (%)
            - MdR: Median rank of the correct match
            - MnR: Mean rank of the correct match
            - cols: List of per-sample ranks (for debugging/analysis)
    """
    # Sort similarities in descending order for each query.
    sorted_similarities = np.sort(-similarity_matrix, axis=1)
    diag_scores = np.diag(-similarity_matrix)[:, np.newaxis]
    rank_indices = (sorted_similarities - diag_scores)
    rank_indices = np.where(rank_indices == 0)
    rank_indices = rank_indices[1]  # Rank of the ground-truth match for each query.

    result_metrics = {}
    result_metrics['R1'] = float(np.sum(rank_indices == 0)) * 100 / len(rank_indices)
    result_metrics['R5'] = float(np.sum(rank_indices < 5)) * 100 / len(rank_indices)
    result_metrics['R10'] = float(np.sum(rank_indices < 10)) * 100 / len(rank_indices)
    result_metrics['Rsum'] = result_metrics['R1'] + result_metrics['R5'] + result_metrics['R10']
    result_metrics['R50'] = float(np.sum(rank_indices < 50)) * 100 / len(rank_indices)
    result_metrics['MdR'] = np.median(rank_indices) + 1
    result_metrics["MnR"] = np.mean(rank_indices) + 1
    result_metrics["cols"] = [int(i) for i in list(rank_indices)]
    return result_metrics


def get_metrics(similarity, n_):
    """
    Wrap metrics() and return a flat list suitable for PrettyTable formatting.

    Args:
        similarity (numpy.ndarray): Similarity matrix.
        n_ (str): Row label for the table.

    Returns:
        list: [label, R1, R5, R10, RSum, MdR, MnR].
    """
    p2p_metrics = metrics(similarity)
    r1, r5, r10, rsum, mdr, mnr = (
        p2p_metrics['R1'], p2p_metrics['R5'], p2p_metrics['R10'],
        p2p_metrics['Rsum'], p2p_metrics['MdR'], p2p_metrics['MnR'],
    )
    return [n_, r1, r5, r10, rsum, mdr, mnr]


class Evaluator:
    """
    Evaluate a ReID model on the test set and log ranking metrics.

    This class extracts L2-normalized image embeddings from the test loader
    and computes similarity-based ranking statistics.

    Args:
        test_loader (DataLoader): DataLoader providing (id, tar_img, can_img) batches.
    """

    def __init__(self, test_loader):
        self.test_loader = test_loader
        self.logger = logging.getLogger("reid")

    def _compute_embedding(self, model):
        """
        Extract L2-normalized embeddings for all test samples.

        Args:
            model (nn.Module): The ReID model (in eval mode).

        Returns:
            tuple: (id_list, tar_feat_list, can_feat_list)
                - id_list (numpy.ndarray): Array of person IDs, shape [N].
                - tar_feat_list (Tensor): Stacked target features, shape [N, D].
                - can_feat_list (Tensor): Stacked candidate features, shape [N, D].
        """
        model = model.eval()
        device = next(model.parameters()).device

        id_list, tar_feat_list, can_feat_list = [], [], []

        for n_iter, batch in tqdm(enumerate(self.test_loader), desc="extracting embeddings"):
            batch_ids = batch['id']
            tar_img = batch['tar_img'].to(device)
            can_img = batch['can_img'].to(device)
            with torch.no_grad():
                tar_feat = model.encode_image(tar_img)
                can_feat = model.encode_image(can_img)
                tar_feat = tar_feat / tar_feat.norm(dim=-1, keepdim=True)
                can_feat = can_feat / can_feat.norm(dim=-1, keepdim=True)

            id_list.extend(batch_ids)
            tar_feat_list.append(tar_feat)
            can_feat_list.append(can_feat)

        id_list = np.array(id_list)
        tar_feat_list = torch.cat(tar_feat_list, dim=0)
        can_feat_list = torch.cat(can_feat_list, dim=0)

        return id_list, tar_feat_list.cpu(), can_feat_list.cpu()

    def eval(self, model):
        """
        Compute the similarity matrix and report ranking metrics.

        Args:
            model (nn.Module): The ReID model.

        Returns:
            float: The RSum metric (used for best checkpoint selection).
        """
        _, tar_feat_list, can_feat_list = self._compute_embedding(model)

        # L2-normalize again on CPU (belt-and-suspenders).
        tar_feat_list = tar_feat_list / tar_feat_list.norm(dim=-1, keepdim=True)
        can_feat_list = can_feat_list / can_feat_list.norm(dim=-1, keepdim=True)
        sims = tar_feat_list @ can_feat_list.t()

        sims_dict = {
            'sims': sims,
        }
        table = PrettyTable(["item", "R1", "R5", "R10", "RSum", "MdR", "MnR"])
        for key in sims_dict.keys():
            sims = sims_dict[key]
            row = get_metrics(sims, f'{key}')
            table.add_row(row)

        # Format numeric columns to one decimal place for readability.
        for col in ["R1", "R5", "R10", "RSum", "MnR", "MdR"]:
            table.custom_format[col] = lambda _, v: f"{v:.1f}"
        self.logger.info('\n' + str(table))

        return row[4]  # RSum
