import torch
import torch.nn.functional as F

def combine_separate_knns(knn_im2im, sim_im2im, knn_im2text, sim_im2text, num_classes):
    """Combines KNNs and similarities from image-image and image-text relations."""
    knn_im = knn_im2im + num_classes
    sim_im = sim_im2im

    knn = torch.cat((knn_im, knn_im2text), dim=1)
    sim = torch.cat((sim_im, sim_im2text), dim=1)

    return knn, sim

def combine_separate_knns_fewshot(knn_im2im, sim_im2im, knn_im2text, sim_im2text, knn_fewshot, sim_fewshot,
                                    num_classes, num_fewshot):
    """Combines KNNs and similarities from image-image, image-text and image-fewshot relations."""
    knn_fewshot = knn_fewshot + num_classes
    knn_im = knn_im2im + num_classes + num_fewshot
    sim_im = sim_im2im

    knn = torch.cat((knn_im, knn_im2text, knn_fewshot), dim=1)
    sim = torch.cat((sim_im, sim_im2text, sim_fewshot), dim=1)

    return knn, sim

def sparse_slice(sparse_matrix, row_start, row_end):
    """Slices a sparse matrix along the row dimension."""
    indices = sparse_matrix.indices()
    values = sparse_matrix.values()

    # Filter row indices within the range [row_start, row_end)
    mask = (indices[0] >= row_start) & (indices[0] < row_end)
    new_indices = indices[:, mask]
    new_values = values[mask]

    # Adjust row indices to match the new submatrix size
    new_indices[0] = new_indices[0] - row_start

    # Create the new sparse matrix
    new_size = (row_end - row_start, sparse_matrix.size(1))
    new_sparse_matrix = torch.sparse_coo_tensor(new_indices, new_values, new_size)

    return new_sparse_matrix


def edge_reweighting(X, Q, a, k, return_full=False, force_1=False):
    """
    Calculates weighted similarity between two sets of feature vectors.

    Args:
        X (torch.Tensor): Feature vectors of shape (N, D).
        Q (torch.Tensor): Feature vectors of shape (M, D).
        a (torch.Tensor): Another set of feature vectors of shape (A, D) used for weighting.
        k (int): Number of top-k neighbors to return.
        return_full (bool, optional): Whether to return full similarity matrix. Defaults to False.
        force_1 (bool, optional): Force to use weights calculated from `a` only. Defaults to False.

    Returns:
        torch.Tensor: Top-k nearest neighbor indices and similarities if return_full is False,
                       else the full similarity matrix.
    """

    # Calculate feature weights based on variance across dimensions
    

    if (X.shape != Q.shape) and (not force_1):
        # Alternative weighting scheme when shapes differ

        weights = 1 / (X.view(1, -1, a.shape[1]).var(dim=1).mean(dim=0))
        Q_weighted = Q * weights
        X_weighted = X 
    else:
        # Use the first weighting scheme when shapes are the same
        weights = a.var(dim=0)
        weights = F.normalize(weights, p=2, dim=0)
        Q_weighted = Q * weights
        X_weighted = X

    # Normalize weighted feature vectors
    Q_weighted = F.normalize(Q_weighted, p=2, dim=1)
    X_weighted = F.normalize(X_weighted, p=2, dim=1)

    # Calculate weighted similarity matrix
    weighted_similarity = Q_weighted @ X_weighted.t()

    if return_full:
        return weighted_similarity

    # Find top-k nearest neighbors
    s, knn = weighted_similarity.topk(k, largest=True, dim=1)

    return knn, s


def knn_to_weighted_adj_matrix(knn, s, num_nodes):
    """
    Converts k-nearest neighbor indices and similarities to a sparse weighted adjacency matrix.

    Args:
        knn (torch.Tensor): k-nearest neighbor indices of shape (num_nodes, k).
        s (torch.Tensor): Similarities corresponding to the k-nearest neighbors of shape (num_nodes, k).
        num_nodes (int): Total number of nodes in the graph.

    Returns:
        torch.sparse_coo_tensor: Sparse weighted adjacency matrix.
    """

    # Flatten the knn and s arrays
    knn_indices = knn.flatten()
    distances = s.flatten()

    # Replace -1 indices (padding) with 0
    knn_indices[knn_indices == -1] = 0

    # Create row indices
    row_indices = torch.arange(num_nodes, device=knn.device).repeat_interleave(knn.shape[1])

    # Filter out zero distances
    nonzero_mask = distances != 0
    filtered_row_indices = row_indices[nonzero_mask]
    filtered_knn_indices = knn_indices[nonzero_mask]
    filtered_distances = distances[nonzero_mask]

    # Create the sparse weighted adjacency matrix
    W = torch.sparse_coo_tensor(
        torch.stack([filtered_row_indices, filtered_knn_indices]),
        filtered_distances,
        (num_nodes, num_nodes)
    )

    return W