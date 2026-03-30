import torch
from pytorch3d.ops import corresponding_points_alignment
from torch.utils.checkpoint import checkpoint
from core.COG.utils.model_utils import pairwise_distance


def _update_once(log_K, u, v, log_W_row, log_W_col):
    u_new = log_W_row - torch.logsumexp(log_K + v.unsqueeze(1), dim=-1)
    v_new = log_W_col - torch.logsumexp(log_K.transpose(-2, -1) + u_new.unsqueeze(1), dim=-1)
    return u_new, v_new


def sinkhorn_weight_prior(similarity, W_row, W_col, num_iters=10, temperature=0.1,
                    normalize=True, use_checkpoint=False,
                    prior=None, prior_weight=0.0, min_prior=1e-12):
    """
    Generalized Sinkhorn in log-domain (stable) with semantic prior P0:
        objective ~ <C, P> + eps * KL(P || a b^T) + lambda * KL(P || P0)
    where kernel K = exp(sim/eps) * P0^{lambda/eps}.

    Args:
        similarity: (B, N, M)  // larger = better
        W_row: (B, N)          // desired row sums (a)
        W_col: (B, M)          // desired col sums (b)
        num_iters: int
        temperature: float     // eps
        normalize: bool        // normalize W_row/W_col to comparable scales
        prior: (B, N, M) or None  // semantic prior P0 (need >0)
        prior_weight: float    // lambda, strength of KL(P||P0)
        min_prior: float       // floor for prior to avoid -inf when taking log

    Returns:
        P: (B, N, M) transport matrix
    """
    B, N, M = similarity.shape
    eps = 1e-9

    # ----- kernel in log-domain -----
    # base kernel from similarity
    log_K = similarity / temperature  # (B, N, M)

    # inject prior via multiplicative factor: K *= prior^(lambda/temperature)
    if prior is not None and prior_weight != 0.0:
        # ensure positivity to keep log well-defined
        prior = prior.clamp_min(min_prior)
        log_K = log_K + (prior_weight / temperature) * torch.log(prior)

    # ----- normalize marginals if requested -----
    if normalize:
        row_mean = W_row.mean(dim=1, keepdim=True)  # (B,1)
        col_mean = W_col.mean(dim=1, keepdim=True)  # (B,1)
        W_row_normalized = W_row / (row_mean + eps)
        W_col_normalized = W_col / (col_mean + eps)
    else:
        W_row_normalized = W_row
        W_col_normalized = W_col

    log_W_row = torch.log(W_row_normalized + eps)  # (B, N)
    log_W_col = torch.log(W_col_normalized + eps)  # (B, M)

    # ----- Sinkhorn iterations in log-domain -----
    u = torch.zeros(B, N, device=similarity.device)
    v = torch.zeros(B, M, device=similarity.device)

    # iterate
    if use_checkpoint:
        for _ in range(num_iters):
            u, v = checkpoint(_update_once, log_K, u, v, log_W_row, log_W_col)
    else:
        for _ in range(num_iters):
            u = log_W_row - torch.logsumexp(log_K + v.unsqueeze(1), dim=-1)  # (B, N)
            v = log_W_col - torch.logsumexp(log_K.transpose(-2, -1) + u.unsqueeze(1), dim=-1)  # (B, M)

    log_P = log_K + u.unsqueeze(-1) + v.unsqueeze(1)
    P = torch.exp(log_P)  # (B, N, M)
    return P


def umeyama_bidirectional(p1, p1_corr, p2, p2_corr, c1, c2, estimate_scale=False, add_noise_eps=1e-5):
    """
    Umeyama algorithm for bidirectional correspondences.
    R,t,s = Umeyama([p1, p1_corr], [p2_corr, p2]) with weights [w1, w2].
    Args:
        p1: (B, N, 3) source point cloud
        p1_corr: (B, M, 3) corresponding points in target for p1
        p2: (B, M, 3) target point cloud
        p2_corr: (B, N, 3) corresponding points in source for p2
        c1: (B, N) weight scores for points in p1
        c2: (B, M) weight scores for points in p2
        estimate_scale: whether to estimate scale factor
        add_noise_eps: small noise to add to points to avoid singularities
    Returns:
        R: (B, 3, 3) rotation matrices
        t: (B, 3) translation vectors
        s: (B,) scale factors
    """

    p1_concat = torch.cat([p1, p1_corr], dim=1)  # (B, N+M, 3)
    p2_concat = torch.cat([p2_corr, p2], dim=1)  # (B, N+M, 3)
    w1 = c1 / (c1.mean(dim=-1, keepdim=True) + 1e-6)
    w2 = c2 / (c2.mean(dim=-1, keepdim=True) + 1e-6)
    w_concat = torch.cat([w1, w2], dim=1)  # (B, N+M)
    p_noise = p1_concat + torch.randn_like(p1_concat) * add_noise_eps  # Add noise to p to avoid singularities
    q_noise = p2_concat + torch.randn_like(p2_concat) * add_noise_eps  # Add noise to q to avoid singularities
    R, t, s = corresponding_points_alignment(p_noise, q_noise, weights=w_concat, estimate_scale=estimate_scale)
    return R, t, s


def apply_transformation(points, R, t, s=None):
    """
    Apply rotation, translation and scaling to points.
    transformed_points = s * (points @ R) + t
    Args:
        points: (B, N, 3) points to transform
        R: (B, 3, 3) rotation matrices
        t: (B, 3) translation vectors
        s: (B,) scale factors
    
    Returns:
        transformed_points: (B, N, 3) transformed points
    """
    if s is None:
        s = torch.ones(points.shape[0], device=points.device)
    transformed_points = s.reshape(-1, 1, 1) * torch.matmul(points, R) + t.reshape(-1, 1, 3)
    return transformed_points


def inverse_transformation(R, t, s):
    """
    Compute the inverse of a rigid transformation (rotation, translation, scale).
    s_inv = 1/s
    R_inv = R^T
    t_inv = -s_inv * R^T * t
    Args:
        R: (B, 3, 3) rotation matrices
        t: (B, 3) translation vectors  
        s: (B,) scale factors
    
    Returns:
        R_inv: (B, 3, 3) inverse rotation matrices
        t_inv: (B, 3) inverse translation vectors
        s_inv: (B,) inverse scale factors
    """
    # Inverse rotation: R^(-1) = R^T for rotation matrices
    R_inv = R.transpose(-2, -1)  # (B, 3, 3)
    
    # Inverse scale: s^(-1) = 1/s
    s_inv = 1.0 / (s + 1e-8)  # Add small epsilon to avoid division by zero
    
    # Inverse translation: t_inv = -s_inv * R^T * t
    # This accounts for the order of operations: first undo translation, then undo rotation and scale
    t_inv = -s_inv.unsqueeze(-1) * torch.bmm(t.unsqueeze(1), R_inv).squeeze(1)  # (B, 3)
    
    return R_inv, t_inv, s_inv
