import torch
import torch.nn.functional as F
import math
import torch.nn as nn
import warnings

"""
Patch Token Compression and Clustering Components for TSI-TVR.

Core Concepts:
  token_dict is the core data structure used throughout this module to represent
  tokens at any stage. It is a dict with the following keys:
    - x (Tensor[B, N, C]): feature tensor of tokens, where B is batch size,
      N is the number of tokens, and C is the feature dimension.
    - token_num (int): current number of valid tokens (i.e., N).
    - idx_token (Tensor[B, N_init]): mapping from the initial grid (with size
      H_init * W_init given by init_grid_size) to current token indices. For
      example, if an initial position is assigned to the i-th token, its value
      is i. N_init is the initial grid size.
    - agg_weight (Tensor[B, N_init, 1]): weight of each initial grid position
      when aggregating to the corresponding token. Used for weighted averaging
      in map2token / token_downup operations.
    - mask (Tensor[B, N] or None): token mask where non-zero means the token is
      valid and zero means it is a padded empty token. If None, all tokens are
      considered valid.

Key Functions:
  - token2map / map2token: Bidirectional conversion between token features and 2D feature maps.
  - token_downup: Transform features between different token distributions.
  - cluster_dpc_knn: Dynamic token clustering based on DPC-KNN
    (Density Peak Clustering with K-Nearest Neighbors).
  - merge_tokens: Merge clustered tokens into cluster center representations.
  - PCM (Progressive Clustering Module): Core clustering module that computes token scores
    and then calls DPC-KNN for downsampling.
  - Att_PCM: Attention mechanism based on PCM, supporting spatial reduction (sr_ratio).
  - Att_Block_Patch: Transformer Block using Att_PCM.
  - TokenConv: 1D convolution on token sequences.
"""


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """
    Fill the input Tensor with values drawn from a truncated normal distribution.
    This is a helper function without gradient tracking.

    The method uses a truncated uniform distribution and then applies the
    inverse CDF (quantile function) of the normal distribution to obtain
    truncated standard normal samples.

    Args:
        tensor (Tensor): an n-dimensional `torch.Tensor` to be filled in-place.
        mean (float): the mean of the normal distribution.
        std (float): the standard deviation of the normal distribution.
        a (float): the minimum cutoff value.
        b (float): the maximum cutoff value.

    Returns:
        Tensor: the input tensor after filling.
    """
    # Compute the cumulative distribution function (CDF) of standard normal distribution
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Step 1: Sample from truncated uniform distribution, map to [2l-1, 2u-1]
        # where l and u are the CDF values corresponding to the truncation interval [a, b]
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Step 2: Use inverse error function (erfinv) to convert uniform distribution to standard normal
        tensor.erfinv_()

        # Step 3: Scale to target mean and standard deviation
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Step 4: Finally clamp values to [a, b] to ensure strict truncation condition
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Fills the input Tensor with values drawn from a truncated normal distribution.
    The values are effectively drawn from the normal distribution
    :math:`\\mathcal{N}(\\text{mean}, \\text{std}^2)` with values outside
    :math:`[a, b]` redrawn until they are within the bounds. The method used
    for generating the random values works best when :math:`a \\leq \\text{mean} \\leq b`.

    Args:
        tensor: an n-dimensional `torch.Tensor`.
        mean: the mean of the normal distribution.
        std: the standard deviation of the normal distribution.
        a: the minimum cutoff value.
        b: the maximum cutoff value.

    Returns:
        Tensor: the input tensor after filling.
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample.

    Randomly drops entire sample paths during training based on the given drop
    probability. This is commonly applied in the main path of residual blocks
    to regularize deep networks.

    Args:
        x (Tensor): input tensor of arbitrary shape.
        drop_prob (float): probability of dropping a path, in [0, 1].
        training (bool): whether in training mode. If False, no dropping is applied.

    Returns:
        Tensor: output tensor after applying stochastic depth.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # Construct broadcast-compatible shape for various tensor dimensions (not limited to 2D ConvNets)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # Binarization: retain with probability keep_prob
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This module wraps the `drop_path` function to provide a convenient nn.Module
    interface that automatically respects `self.training` mode.
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def get_grid_index(init_size, map_size, device):
    """Get the index of each initial grid in the flattened feature map.

    Maps each position from the initial grid resolution to the nearest position
    in the target feature map resolution via nearest-neighbor interpolation.

    Args:
        init_size (list[int] or tuple[int]): initial grid resolution [H_init, W_init].
        map_size (list[int] or tuple[int]): feature map resolution [H, W].
        device: torch device of the output tensor.

    Returns:
        Tensor[H_init * W_init]: flattened indices in the feature map for each initial grid position.
    """
    H_init, W_init = init_size
    H, W = map_size
    # Generate absolute indices on the feature map, shape [1, 1, H, W]
    idx = torch.arange(H * W, device=device).reshape(1, 1, H, W)
    # Map indices to initial grid resolution via nearest-neighbor interpolation
    idx = F.interpolate(idx.float(), [H_init, W_init], mode='nearest').long()
    return idx.flatten()


def index_points(points, idx):
    """Sample features following the given index.

    Gathers points along the token (point) dimension using batch-wise indices.

    Args:
        points (Tensor[B, N, C]): input points data.
        idx (Tensor[B, S]): sample index data, each row contains S indices in [0, N-1].

    Returns:
        Tensor[B, S, C]: indexed points data.
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    # Generate batch dimension indices matching idx shape for broadcasting
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def token2map(token_dict):
    """Transform vision tokens to a 2D feature map.

    This function reconstructs a feature map from token features using the
    mapping information stored in `token_dict`. It supports efficient sparse
    or dense matrix multiplication depending on which path has fewer FLOPs.

    Note:
        This function only works when the resolution of the target feature map
        is not higher than the initial grid structure.

    Args:
        token_dict (dict): token information dict, must contain:
            - x (Tensor[B, N, C]): token features.
            - map_size (list[int]): target feature map resolution [H, W].
            - init_grid_size (list[int]): initial grid resolution [H_init, W_init].
            - idx_token (Tensor[B, N_init]): mapping from initial grid to token index.

    Returns:
        Tensor[B, C, H, W]: reconstructed feature map.
    """
    x = token_dict['x']
    H, W = token_dict['map_size']
    H_init, W_init = token_dict['init_grid_size']
    idx_token = token_dict['idx_token']
    B, N, C = x.shape
    N_init = H_init * W_init
    device = x.device

    if N_init == N and N == H * W:
        # Initial tokens have full grid structure, just reshape
        return x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

    # For each initial grid position, get its corresponding index in the flattened feature map
    idx_hw = get_grid_index(
        [H_init, W_init], [H, W], device=device)[None, :].expand(B, -1)
    idx_batch = torch.arange(B, device=device)[:, None].expand(B, N_init)
    value = x.new_ones(B * N_init)

    # Choose sparse or dense matrix multiplication based on computational complexity
    if N_init < N * H * W:
        # When N_init is small, sparse matrix multiplication is more efficient
        # Computational cost is approximately B * N_init * (C+2)
        idx_hw = idx_hw + idx_batch * H * W
        idx_tokens = idx_token + idx_batch * N
        coor = torch.stack([idx_hw, idx_tokens], dim=0).reshape(2, B * N_init)

        # torch.sparse does not support fp16, temporarily switch precision
        with torch.cuda.amp.autocast(enabled=False):
            # torch.sparse does not support gradient for sparse tensors, so detach is needed
            value = value.detach().float()

            # Build sparse matrix A with shape [B * H * W, B * N]
            A = torch.sparse.FloatTensor(coor, value, torch.Size([B * H * W, B * N]))

            # Normalize each row so each feature map position is the weighted average of its token
            all_weight = A @ x.new_ones(B * N, 1).type(torch.float32) + 1e-6
            value = value / all_weight[idx_hw.reshape(-1), 0]

            # Update sparse matrix with normalized weights
            A = torch.sparse.FloatTensor(coor, value, torch.Size([B * H * W, B * N]))

            # Sparse matrix multiplication: aggregate token features to feature map positions
            x_out = A @ x.reshape(B * N, C).type(torch.float32)  # [B*H*W, C]

    else:
        # When N_init is large, dense matrix multiplication is more efficient
        # Computational cost is approximately B * N * H * W * (C+2)
        coor = torch.stack([idx_batch, idx_hw, idx_token], dim=0).reshape(3, B * N_init)

        # Build dense matrix A with shape [B, H*W, N]
        A = torch.sparse.FloatTensor(coor, value, torch.Size([B, H * W, N])).to_dense()
        # Normalize each row
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)

        x_out = A @ x  # [B, H*W, C]

    x_out = x_out.type(x.dtype)
    x_out = x_out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
    return x_out


def map2token(feature_map, token_dict):
    """Transform a 2D feature map to vision tokens.

    This is the inverse operation of `token2map`. It samples features from the
    feature map back to token positions based on the mapping in `token_dict`.

    Note:
        This function only works when the feature map resolution is not higher
        than the initial grid structure.

    Args:
        feature_map (Tensor[B, C, H, W]): input feature map.
        token_dict (dict): token information dict, must contain:
            - idx_token (Tensor[B, N_init]): mapping from initial grid to token index.
            - token_num (int): number of tokens N.
            - init_grid_size (list[int]): initial grid resolution [H_init, W_init].

    Returns:
        Tensor[B, N, C]: token features sampled from the feature map.
    """
    idx_token = token_dict['idx_token']
    N = token_dict['token_num']
    H_init, W_init = token_dict['init_grid_size']
    N_init = H_init * W_init

    # agg_weight can be used for weighted sampling, but is not enabled in current implementation
    agg_weight = None

    B, C, H, W = feature_map.shape
    device = feature_map.device

    if N_init == N and N == H * W:
        # Initial tokens have full grid structure, just reshape and convert
        return feature_map.flatten(2).permute(0, 2, 1).contiguous()

    # Get the corresponding index of each initial grid position on the feature map
    idx_hw = get_grid_index(
        [H_init, W_init], [H, W], device=device)[None, :].expand(B, -1)

    idx_batch = torch.arange(B, device=device)[:, None].expand(B, N_init)
    if agg_weight is None:
        value = feature_map.new_ones(B * N_init)
    else:
        value = agg_weight.reshape(B * N_init).type(feature_map.dtype)

    # Choose sparse or dense matrix multiplication based on computation cost
    if N_init < N * H * W:
        # Sparse matrix path, computational cost approximately B * N_init * (C+2)
        idx_token = idx_token + idx_batch * N
        idx_hw = idx_hw + idx_batch * H * W
        indices = torch.stack([idx_token, idx_hw], dim=0).reshape(2, -1)

        # torch.sparse does not support fp16
        with torch.cuda.amp.autocast(enabled=False):
            # Sparse matrix does not support gradients, detach needed
            value = value.detach().float()
            # Build sparse matrix A with shape [B*N, B*H*W]
            A = torch.sparse_coo_tensor(indices, value, (B * N, B * H * W))
            # Normalize by row
            all_weight = A @ torch.ones(
                [B * H * W, 1], device=device, dtype=torch.float32) + 1e-6
            value = value / all_weight[idx_token.reshape(-1), 0]

            A = torch.sparse_coo_tensor(indices, value, (B * N, B * H * W))
            # Aggregate feature map features back to token positions
            out = A @ feature_map. \
                permute(0, 2, 3, 1).contiguous().reshape(B * H * W, C).float()
    else:
        # Dense matrix path, computational cost approximately B * N * H * W * (C+2)
        indices = torch.stack([idx_batch, idx_token, idx_hw], dim=0).reshape(3, -1)
        value = value.detach()  # detach to reduce computation cost during training
        A = torch.sparse_coo_tensor(indices, value, (B, N, H * W)).to_dense()
        # Normalize by row
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)

        out = A @ feature_map.permute(0, 2, 3, 1).reshape(B, H * W, C).contiguous()

    out = out.type(feature_map.dtype)
    out = out.reshape(B, N, C)
    return out


def token_downup(target_dict, source_dict):
    """Transform token features between different token distributions.

    Maps token features from the source distribution to the target distribution
    using the index mappings stored in both token dicts. This is useful when
    tokens have been clustered/merged differently across layers.

    Args:
        target_dict (dict): dict for target token information.
        source_dict (dict): dict for source token information.

    Returns:
        Tensor[B, T, C]: transformed token features, where T is the target token number.
    """
    x_s = source_dict['x']
    idx_token_s = source_dict['idx_token']
    idx_token_t = target_dict['idx_token']
    T = target_dict['token_num']
    B, S, C = x_s.shape
    N_init = idx_token_s.shape[1]

    # Use aggregation weight from target_dict, default to all ones if not present
    weight = target_dict['agg_weight'] if 'agg_weight' in target_dict.keys() else None
    if weight is None:
        weight = x_s.new_ones(B, N_init, 1)
    weight = weight.reshape(-1)

    # Choose sparse or dense matrix multiplication based on computation cost
    if N_init < T * S:
        # Sparse matrix path, computational cost approximately B * N_init * (C+2)
        # Expand token indices to globally unique indices (across batch)
        idx_token_t = idx_token_t + torch.arange(B, device=x_s.device)[:, None] * T
        idx_token_s = idx_token_s + torch.arange(B, device=x_s.device)[:, None] * S
        coor = torch.stack([idx_token_t, idx_token_s], dim=0).reshape(2, B * N_init)

        # torch.sparse.spmm does not support fp16
        with torch.cuda.amp.autocast(enabled=False):
            # Sparse matrix does not support gradients, detach needed
            weight = weight.float().detach()
            # Build sparse matrix A with shape [B*T, B*S]
            A = torch.sparse.FloatTensor(coor, weight, torch.Size([B * T, B * S]))
            # Normalize by row
            all_weight = A.type(torch.float32) @ x_s.new_ones(B * S, 1).type(torch.float32) + 1e-6
            weight = weight / all_weight[(idx_token_t).reshape(-1), 0]
            A = torch.sparse.FloatTensor(coor, weight, torch.Size([B * T, B * S]))
            # Sparse matrix multiplication to complete distribution transformation
            x_out = A.type(torch.float32) @ x_s.reshape(B * S, C).type(torch.float32)
    else:
        # Dense matrix path, computational cost approximately B * T * S * (C+2)
        idx_batch = torch.arange(B, device=x_s.device)[:, None].expand(B, N_init)
        coor = torch.stack([idx_batch, idx_token_t, idx_token_s], dim=0).reshape(3, B * N_init)
        weight = weight.detach()  # detach to reduce training time
        # Build dense matrix A with shape [B, T, S]
        A = torch.sparse.FloatTensor(coor, weight, torch.Size([B, T, S])).to_dense()
        # Normalize by row
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)
        # Dense matrix multiplication to complete distribution transformation
        x_out = A @ x_s

    x_out = x_out.reshape(B, T, C).type(x_s.dtype)
    return x_out


def cluster_dpc_knn(token_dict, cluster_num, k=5, token_mask=None):
    """Cluster tokens using the DPC-KNN (Density Peak Clustering with KNN) algorithm.

    This algorithm clusters tokens based on local density and distance to higher-density
    points. It is adapted from the Density Peak Clustering (DPC) method, using k-nearest
    neighbors to estimate local density.

    Algorithm steps:
        1. Compute pairwise distance matrix between tokens.
        2. Compute local density for each token using its k nearest neighbors.
        3. Compute the minimum distance from each token to any token with higher density.
        4. Select cluster centers as tokens with the highest (density * distance) scores.
        5. Assign each non-center token to the nearest cluster center.
        6. Ensure each cluster center maps to itself.

    Args:
        token_dict (dict): token information dict, must contain:
            - x (Tensor[B, N, C]): token features.
        cluster_num (int): desired number of clusters.
        k (int): number of nearest neighbors used for local density estimation.
        token_mask (Tensor[B, N] or None): mask indicating valid tokens.
            Non-zero values mean the token is meaningful; zero means padded empty token.
            If None, all tokens are considered meaningful.

    Returns:
        tuple:
            - idx_cluster (Tensor[B, N]): cluster index for each token.
            - cluster_num (int): actual cluster number (same as input).
    """
    with torch.no_grad():
        x = token_dict["x"]
        B, N, C = x.shape

        # Step 1: Compute pairwise Euclidean distance matrix between tokens, normalized by sqrt(feature_dim)
        dist_matrix = torch.cdist(x, x) / (C ** 0.5)

        if token_mask is not None:
            # Set distances for invalid tokens to extremely large values so they do not affect clustering of valid tokens
            token_mask = token_mask > 0
            dist_matrix = dist_matrix * token_mask[:, None, :] + (dist_matrix.max() + 1) * (~token_mask[:, None, :])

        # Step 2: Compute local density
        # For each token, take its k nearest neighbors, density defined as exp(-mean(dist^2))
        dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)
        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
        # Add tiny noise to avoid identical density values, ensuring stable sorting
        density = density + torch.rand(
            density.shape, device=density.device, dtype=density.dtype) * 1e-6

        if token_mask is not None:
            # Force density of invalid tokens to 0 to prevent them from being selected as centers
            density = density * token_mask

        # Step 3: Compute distance indicator
        # For each token, find all tokens with higher density and take the nearest one
        # mask[i,j] = 1 means token j has higher density than token i
        mask = density[:, None, :] > density[:, :, None]
        mask = mask.type(x.dtype)
        dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
        # For tokens with density not higher, replace distance with max value to be ignored in min operation
        dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

        # Step 4: Select cluster centers based on score = density * distance
        # Tokens with high scores have both large density and large distance, matching density peak characteristics
        score = dist * density
        _, index_down = torch.topk(score, k=cluster_num, dim=-1)

        # Step 5: Assign each token to its nearest cluster center
        dist_matrix = index_points(dist_matrix, index_down)
        idx_cluster = dist_matrix.argmin(dim=1)

        # Step 6: Ensure each cluster center merges to itself, preventing misassignment to other clusters
        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
        idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
        idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    return idx_cluster, cluster_num


def merge_tokens(token_dict, idx_cluster, cluster_num, token_weight=None):
    """Merge tokens belonging to the same cluster into a single cluster center.

    This function aggregates token features within each cluster using weighted averaging.
    It updates the token dict to reflect the new merged tokens and their mappings.

    Implementation uses `torch.index_add_()` for efficient aggregation.
    Computational cost is approximately B*N*(C+2).

    Args:
        token_dict (dict): input token information dict, must contain:
            - x (Tensor[B, N, C]): token features.
            - idx_token (Tensor[B, N_init]): mapping from initial grid to token index.
            - agg_weight (Tensor[B, N_init, 1]): aggregation weights.
        idx_cluster (Tensor[B, N]): cluster index assigned to each token.
        cluster_num (int): number of clusters.
        token_weight (Tensor[B, N, 1] or None): weight for each token during merging.
            If None, uniform weights are used.

    Returns:
        dict: output token information dict with merged tokens, containing:
            - x (Tensor[B, cluster_num, C]): merged cluster features.
            - token_num (int): number of merged tokens (cluster_num).
            - idx_token (Tensor): updated mapping from initial grid to merged token index.
            - agg_weight (Tensor): updated aggregation weights.
            - mask (Tensor or None): updated mask for merged tokens.
    """
    x = token_dict['x']
    idx_token = token_dict['idx_token']
    agg_weight = token_dict['agg_weight']

    B, N, C = x.shape
    if token_weight is None:
        token_weight = x.new_ones(B, N, 1)

    # Construct global indices, flatten 2D [B, N] to 1D for index_add_ operation
    idx_batch = torch.arange(B, device=x.device)[:, None]
    idx = idx_cluster + idx_batch * cluster_num

    # Compute total weight of each cluster for normalization
    all_weight = token_weight.new_zeros(B * cluster_num, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N), source=token_weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6
    norm_weight = token_weight / all_weight[idx]

    # Weighted aggregation of token features to corresponding cluster centers
    x_merged = x.new_zeros(B * cluster_num, C)
    source = x * norm_weight
    x_merged.index_add_(dim=0, index=idx.reshape(B * N), source=source.reshape(B * N, C).type(x.dtype))
    x_merged = x_merged.reshape(B, cluster_num, C)

    # Update idx_token: each initial grid position maps to the merged token index of its cluster
    idx_token_new = index_points(idx_cluster[..., None], idx_token).squeeze(-1)
    # Update agg_weight: new aggregation weight is original weight times normalization weight
    weight_t = index_points(norm_weight, idx_token)
    agg_weight_new = agg_weight * weight_t
    agg_weight_new / agg_weight_new.max(dim=1, keepdim=True)[0]

    # If mask exists, sum within each cluster, cluster is valid if sum > 0
    if 'mask' in token_dict and token_dict['mask'] is not None:
        mask = token_dict['mask'].float()
        all_mask = mask.new_zeros(B * cluster_num)
        all_mask.index_add_(
            dim=0,
            index=idx.reshape(B * N),
            source=mask.reshape(B * N)
        )
        mask_new = (all_mask > 0).float().reshape(B, cluster_num)
    else:
        mask_new = None

    out_dict = {}
    out_dict['x'] = x_merged
    out_dict['token_num'] = cluster_num
    out_dict['idx_token'] = idx_token_new
    out_dict['agg_weight'] = agg_weight_new
    out_dict['mask'] = mask_new
    return out_dict


def vis_tokens(img, token_dict, edge_color=[1.0, 1.0, 1.0], edge_width=1):
    """Visualize tokens by drawing boundaries between different token regions.

    This function reconstructs a visualization image where each token region is
    filled with the average color of the corresponding image patch, and edges
    between different tokens are highlighted.

    Args:
        img (Tensor[B, 3, H, W]): input image.
        token_dict (dict): token information dict.
        edge_color (list[float]): RGB color for edges, default is white [1, 1, 1].
        edge_width (int): width of the boundary edges.

    Returns:
        Tensor[B, 3, H*8, W*8]: visualization result with token boundaries.
    """
    N = token_dict['token_num']
    device, dtype = img.device, img.dtype

    # Downsample input image as the base color for each token region
    color_map = F.avg_pool2d(img, kernel_size=4)
    B, C, H, W = color_map.shape

    # Convert color map to token representation, then map back to grid via token2map
    token_color = map2token(color_map, token_dict)
    tmp_dict = token_dict.copy()
    tmp_dict['map_size'] = [H, W]
    tmp_dict['x'] = token_color
    vis_img = token2map(tmp_dict)

    # Generate index map for each token, different tokens have different index values
    token_idx = torch.arange(N, device=device)[None, :, None].float() / N
    tmp_dict['x'] = token_idx
    idx_map = token2map(tmp_dict)  # [B, 1, H, W]

    # Upsample to higher resolution for better display detail
    vis_img = F.interpolate(vis_img, [H * 8, W * 8], mode='nearest')
    idx_map = F.interpolate(idx_map, [H * 8, W * 8], mode='nearest')

    # Define edge detection kernels (up, down, left, right directions)
    kernel = idx_map.new_zeros([4, 1, 3, 3])
    kernel[:, :, 1, 1] = 1
    kernel[0, :, 0, 1] = -1
    kernel[1, :, 2, 1] = -1
    kernel[2, :, 1, 0] = -1
    kernel[3, :, 1, 2] = -1

    # Iterate multiple times to widen edge lines
    for i in range(edge_width):
        edge_map = F.conv2d(F.pad(idx_map, [1, 1, 1, 1], mode='replicate'), kernel)
        edge_map = (edge_map != 0).max(dim=1, keepdim=True)[0]
        idx_map = idx_map * (~edge_map) + torch.rand(idx_map.shape, device=device, dtype=dtype) * edge_map

    # Replace edge positions with specified color
    edge_color = torch.tensor(edge_color, device=device, dtype=dtype)[None, :, None, None]
    vis_img = vis_img * (~edge_map) + edge_color * edge_map
    return vis_img


class TokenConv(nn.Module):
    """1D Convolution module for token sequences.

    Applies a 1D convolution along the feature dimension of token sequences.
    The input is expected in shape [B, N, C], which is permuted to [B, C, N]
    for the Conv1d operation.

    Note:
        This module uses a residual connection: output = input + conv(input).
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, bias=False, padding=0):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size, bias=bias,
                              padding=padding)

    def forward(self, x):
        # Transpose [B, N, C] to [B, C, N] for 1D convolution, then transpose back
        x = x + self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class PCM(nn.Module):
    """Progressive Clustering Module (PCM).

    PCM is the core token compression module. It performs the following steps:
        1. Apply 1D convolution and LayerNorm to token features.
        2. Predict a score for each token using a linear layer.
        3. Convert scores to aggregation weights (with masking for invalid tokens).
        4. Determine the target cluster number based on sample_ratio.
        5. Perform DPC-KNN clustering to obtain cluster assignments.
        6. Merge tokens within each cluster into a single representative token.

    Args:
        sample_ratio (float): ratio of tokens to retain after clustering,
            e.g., 0.5 means compressing to 50% of the original tokens.
        embed_dim (int): input feature dimension of tokens.
        dim_out (int): output feature dimension after convolution.
        k (int): number of nearest neighbors for DPC-KNN density estimation.
    """

    def __init__(self, sample_ratio, embed_dim, dim_out, k=5):
        super().__init__()
        self.sample_ratio = sample_ratio
        self.dim_out = dim_out
    # Use 1D convolution for local transformation of token features
        self.conv = TokenConv(in_channels=embed_dim, out_channels=dim_out, kernel_size=3, bias=False, padding=1)
        self.norm = nn.LayerNorm(self.dim_out)
    # Predict importance score for each token
        self.score = nn.Linear(self.dim_out, 1)
        self.k = k

    def forward(self, token_dict):
        x = token_dict["x"]
        # Step 1: Local feature transformation and normalization
        x = self.conv(x)
        x = self.norm(x)

        # Step 2: Compute token scores and convert to exponential weights
        token_score = self.score(x)
        token_weight = token_score.squeeze(2)
        # Set weight of invalid tokens (mask=0) to negative infinity, making it approach 0 after exp
        if token_dict["mask"] is not None:
            token_weight.masked_fill_((1 - token_dict["mask"]).to(torch.bool), float("-inf"))
        token_weight = token_weight.unsqueeze(2).exp()

        # Update features and scores in token_dict
        token_dict['x'] = x
        B, N, C = x.shape
        token_dict['token_score'] = token_score

        # Step 3: Compute target cluster number based on sample_ratio (keep at least 1 token)
        cluster_num = max(math.ceil(N * self.sample_ratio), 1)

        # Step 4: Call DPC-KNN clustering to get cluster assignment for each token
        idx_cluster, cluster_num = cluster_dpc_knn(
            token_dict, cluster_num, self.k, token_mask=token_dict["mask"])

        # Step 5: Merge tokens by cluster to get downsampled token_dict
        down_dict = merge_tokens(token_dict, idx_cluster, cluster_num, token_weight)
        return down_dict, token_dict


class Att_PCM(nn.Module):
    """Attention module with PCM-based spatial reduction.

    This attention layer supports reducing the spatial resolution of key/value
    tokens through a strided convolution or average pooling when `sr_ratio > 1`.
    It also incorporates token scores from PCM as positional confidence bias
    to the attention scores.

    Args:
        dim (int): total dimension of the model.
        num_heads (int): number of attention heads.
        qkv_bias (bool): whether to add bias to q and kv projections.
        qk_scale (float or None): custom scale factor for QK^T. Defaults to head_dim^-0.5.
        attn_drop (float): dropout rate for attention weights.
        proj_drop (float): dropout rate for the output projection.
        sr_ratio (int): spatial reduction ratio for key/value tokens.
        use_sr_layer (bool): if True, use a Conv2d + LayerNorm for spatial reduction;
            otherwise use average pooling.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, use_sr_layer=True):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Query projection comes from q_dict, Key/Value projections come from kv_dict
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        self.use_sr_layer = use_sr_layer
        # If sr_ratio > 1, perform spatial reduction on kv tokens
        if sr_ratio > 1:
            if self.use_sr_layer:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights for Linear, LayerNorm, and Conv2d modules."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, q_dict, kv_dict):
        """Forward pass of attention with optional spatial reduction.

        Args:
            q_dict (dict): token dict for queries.
            kv_dict (dict): token dict for keys and values.

        Returns:
            Tensor[B, Nq, C]: attended token features.
        """
        q = q_dict['x']
        kv = kv_dict['x']
        B, Nq, C = q.shape
        Nkv = kv.shape[1]
        # Get kv token scores as attention bias, default to 0 if not available
        conf_kv = kv_dict['token_score'] if 'token_score' in kv_dict.keys() else kv.new_zeros(B, Nkv, 1)

        # Project and split multi-head Q
        q = self.q(q).reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        # If spatial reduction needed, first restore kv tokens to feature map, then pool or convolve
        if self.sr_ratio > 1:
            tmp = torch.cat([kv, conf_kv], dim=-1)
            tmp_dict = kv_dict.copy()
            tmp_dict['x'] = tmp
            tmp_dict['map_size'] = q_dict['map_size']
            tmp = token2map(tmp_dict)

            kv = tmp[:, :C]
            conf_kv = tmp[:, C:]

            if self.use_sr_layer:
                # Use convolution + LayerNorm for spatial reduction
                kv = self.sr(kv)
                _, _, h, w = kv.shape
                kv = kv.reshape(B, C, -1).permute(0, 2, 1).contiguous()
                kv = self.norm(kv)
            else:
                # Use average pooling for spatial reduction
                kv = F.avg_pool2d(kv, kernel_size=self.sr_ratio, stride=self.sr_ratio)
                kv = kv.reshape(B, C, -1).permute(0, 2, 1).contiguous()

            conf_kv = F.avg_pool2d(conf_kv, kernel_size=self.sr_ratio, stride=self.sr_ratio)
            conf_kv = conf_kv.reshape(B, 1, -1).permute(0, 2, 1).contiguous()

        # Project and split multi-head K, V
        kv = self.kv(kv).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1]

        # Compute scaled dot-product attention
        attn = (q * self.scale) @ k.transpose(-2, -1)

        # Add token scores as bias to attention logits, enhancing influence of high-confidence tokens
        conf_kv = conf_kv.squeeze(-1)[:, None, None, :]
        attn = attn + conf_kv
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Weighted aggregate Values and project output
        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Att_Block_Patch(nn.Module):
    """Transformer block using Att_PCM for patch tokens.

    This block applies LayerNorm and Att_PCM attention with a residual connection
    and stochastic depth (drop_path).

    Args:
        dim (int): input/output feature dimension.
        num_heads (int): number of attention heads.
        mlp_ratio (float): ratio of MLP hidden dim to input dim (kept for interface consistency).
        qkv_bias (bool): whether to add bias to qkv projections.
        qk_scale (float or None): custom scale for QK^T.
        drop (float): dropout rate for projection.
        attn_drop (float): dropout rate for attention weights.
        drop_path (float): stochastic depth rate.
        act_layer (nn.Module): activation layer class (reserved for interface consistency).
        norm_layer (nn.Module): normalization layer class.
        sr_ratio (int): spatial reduction ratio for kv tokens in Att_PCM.
        use_sr_layer (bool): whether to use Conv2d for spatial reduction in Att_PCM.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, use_sr_layer=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Att_PCM(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, use_sr_layer=use_sr_layer)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights for Linear, LayerNorm, and Conv2d modules."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, inputs):
        """Forward pass of the transformer block.

        Args:
            inputs (tuple): (q_dict, kv_dict) pair of token dicts.

        Returns:
            dict: updated q_dict with attended features.
        """
        q_dict, kv_dict = inputs

        x = q_dict['x']
        # Apply LayerNorm to both q and kv
        q_dict['x'] = self.norm1(q_dict['x'])
        kv_dict['x'] = self.norm1(kv_dict['x'])

        # Residual connection: original features + DropPath(Attention)
        x = x + self.drop_path(self.attn(q_dict, kv_dict))

        q_dict['x'] = x
        return q_dict


if __name__ == '__main__':

    embed_dim = 512

    sr_p = [0.5, 0.5, 0.5]
    v_pcm_p_1 = PCM(sample_ratio=sr_p[0], embed_dim=embed_dim, dim_out=embed_dim, k=3)
    v_att_block_p_1 = Att_Block_Patch(dim=embed_dim, num_heads=8)
    v_pcm_p_2 = PCM(sample_ratio=sr_p[1], embed_dim=embed_dim, dim_out=embed_dim, k=3)
    v_att_block_p_2 = Att_Block_Patch(dim=embed_dim, num_heads=8)
    v_pcm_p_3 = PCM(sample_ratio=sr_p[2], embed_dim=embed_dim, dim_out=embed_dim, k=3)
    v_att_block_p_3 = Att_Block_Patch(dim=embed_dim, num_heads=8)

    p_feat = torch.rand(32, 48, 512)    # (384 * 128 * 3) / (32 * 32 * 3)
    p_idx_token = torch.arange(p_feat.size(1))[None, :].repeat(p_feat.size(0), 1)
    p_agg_weight = p_feat.new_ones(p_feat.size(0), p_feat.size(1), 1)
    p_mask = p_feat.new_ones(p_feat.size(0), p_feat.size(1))
    p_token_dict = {'x': p_feat,
                    'token_num': p_feat.size(1),
                    'idx_token': p_idx_token,
                    'agg_weight': p_agg_weight,
                    'mask': p_mask.detach()}

    p_token_dict = v_att_block_p_1(v_pcm_p_1(p_token_dict))
    p_token_dict = v_att_block_p_2(v_pcm_p_2(p_token_dict))
    p_token_dict = v_att_block_p_3(v_pcm_p_3(p_token_dict))

    p_feat = p_token_dict['x']

    p_feat = p_feat.view(32, -1, 512)
    print(p_feat.shape)