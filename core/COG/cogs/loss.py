import torch
import torch.nn.functional as F
import torch.nn as nn
from pytorch3d.loss import chamfer_distance
from .utils import apply_transformation, inverse_transformation

def compute_cycle_loss(
        end_points,
        p,
        p_reconstructed,
        p_confidence,
        q,
        q_reconstructed,
        q_confidence,
        alpha=50,
        loss_str="cycle",
        loss_weight=1
        ):
    """
    Compute the cycle consistency loss and store in end_points.
    
    Args:
        end_points: dictionary to store loss values
        p: (B, N, 3) original points
        p_reconstructed: (B, N, 3) reconstructed points
        confidence: (B, N) confidence scores for points in p
        alpha: hyperparameter for the Gaussian RBF
        loss_str: prefix for loss key names
        loss_weight: weight for the loss
    
    Returns:
        end_points: updated dictionary with loss values
    """
    lh_p, loss_dis_p = loss_cycle(p, p_reconstructed, p_confidence, alpha)
    lh_q, loss_dis_q = loss_cycle(q, q_reconstructed, q_confidence, alpha)
    loss = (loss_dis_p + loss_dis_q) / 2
    
    # Store loss and distance in end_points
    end_points[f"{loss_str}_loss"] = loss * loss_weight
    #end_points[f"{loss_str}_distance"] = dis_l1
    
    return end_points, lh_p, lh_q


def compute_chamfer_loss(
        end_points,
        p, q, R, t, s,
        p_confidence, q_confidence,
        alpha=50, loss_str="chamfer",
        loss_weight=1
        ):
    """
    Compute the chamfer distance loss and store in end_points.
    
    Args:
        end_points: dictionary to store loss values
        p: (B, N, 3) original points
        q: (B, N, 3) transformed points
        R: (B, 3, 3) rotation matrices
        t: (B, 3) translation vectors
        s: (B,) scale factors
        confidence: (B, N) confidence scores
        alpha: hyperparameter for the Gaussian RBF
        loss_str: prefix for loss key names
        loss_weight: weight for the loss
    
    Returns:
        end_points: updated dictionary with loss values
    """
    lh_p, loss_p = loss_chamfer(p, q, R, t, s, p_confidence, alpha)
    r_inv, t_inv, s_inv = inverse_transformation(R, t, s)
    lh_q, loss_q = loss_chamfer(q, p, r_inv, t_inv, s_inv, q_confidence, alpha)

    loss = (loss_p + loss_q) / 2
    # Store loss and distance in end_points
    end_points[f"{loss_str}_loss"] = loss * loss_weight
    
    return end_points, lh_p, lh_q


def compute_stego_loss(
        end_points,
        d1, d2,
        s1, s2,
        self_bias=0., cross_bias=0.,
        loss_str="stego",
        loss_weight=1
        ):
    """
    Compute the STEGO loss and store in end_points.
    
    Args:
        end_points: dictionary to store loss values
        d1: (B, N, D1) DINO features from point cloud p
        d2: (B, M, D1) DINO features from point cloud q
        s1: (B, N, D2) STEGO features from point cloud p
        s2: (B, M, D2) STEGO features from point cloud q
        bias: bias term, bigger bias encourages less matching
        loss_str: prefix for loss key names
        loss_weight: weight for the loss
    
    Returns:
        end_points: updated dictionary with loss values
    """
    if d1.dim() == 4:
        d1 = d1.reshape(d1.shape[0], d1.shape[1], -1).permute(0,2,1) # B, N, D1
        d2 = d2.reshape(d2.shape[0], d2.shape[1], -1).permute(0,2,1) # b, M, D1
        s1 = s1.reshape(s1.shape[0], s1.shape[1], -1).permute(0,2,1) # b, N, D2
        s2 = s2.reshape(s2.shape[0], s2.shape[1], -1).permute(0,2,1) # b, M, D2
    loss_cross = loss_stego(d1, d2, s1, s2, cross_bias)
    loss_self_1 = loss_stego(d1, d1, s1, s1, self_bias)
    loss_self_2 = loss_stego(d2, d2, s2, s2, self_bias)
    loss_self = (loss_self_1 + loss_self_2) / 2
    loss_value = (loss_cross + loss_self) / 2
    
    # Store loss in end_points
    end_points[f"{loss_str}_loss"] = loss_value * loss_weight
    
    return end_points


def compute_semantic_loss(
        end_points,
        operator_12, operator_21,
        c1, c2,
        sem_sim,
        alpha=2,
        loss_str="semantic",
        loss_weight=1
        ):
    """
    Compute the geo-semantic consistency loss and store in end_points.
    
    Args:
        end_points: dictionary to store loss values
        geo_attn: (B, N, M) geometric attention weights from p to q
        sem_sim: (B, N, M) semantic similarity matrix between p and q
        alpha: hyperparameter for the exponential function
        loss_str: prefix for loss key names
        loss_weight: weight for the loss
    
    Returns:
        end_points: updated dictionary with loss values
    """
    sem_sim_ = sem_sim.detach()  # stop gradient
    semantic_loss_1, sem_lh1 = loss_semantic(operator_12, sem_sim_, c1, alpha)
    semantic_loss_2, sem_lh2 = loss_semantic(operator_21, sem_sim_.transpose(-2, -1), c2, alpha)
    semantic_loss = (semantic_loss_1 + semantic_loss_2) / 2
    
    # Store loss in end_points
    end_points[f"{loss_str}_loss"] = semantic_loss * loss_weight
    
    return end_points, sem_lh1, sem_lh2


def compute_rt_loss(
        end_points,
        p1,
        pred_r,
        pred_t,
        gt_r,
        gt_t,
        alpha=50,
        loss_str="pose",
        loss_weight=1
        ):
    """
    Compute the rotation and translation loss and store in end_points.
    Args:
        end_points: dictionary to store loss values
        p1: (B, N, 3) original points
        pred_r: (B, 3, 3) predicted rotation matrices
        pred_t: (B, 3) predicted translation vectors
        gt_r: (B, 3, 3) ground truth rotation matrices
        gt_t: (B, 3) ground truth translation vectors
        alpha_r: dcd alpha for rotation loss
        alpha_t: dcd alpha for translation loss
        loss_str: prefix for loss key names
        loss_weight: weight for the loss
    Returns:
        end_points: updated dictionary with loss values
    """
    rt_loss = loss_rt(p1, pred_r, pred_t, gt_r, gt_t, alpha).mean()
    end_points[f"{loss_str}_loss"] = rt_loss * loss_weight
    return end_points


def loss_cycle(p, p_reconstructed, confidence, alpha=50):
    """
    Compute the loss between original points p and reconstructed points p_reconstructed.
    The loss is weighted by the confidence scores.
    Args:
        p: (B, N, 3) original points
        p_reconstructed: (B, N, 3) reconstructed points
        p_confidence: (B, N) confidence scores for points in p
        alpha: hyperparameter for the Gaussian RBF
    Returns:
        l1_distance: (B, N) L1 distance between p and p_reconstructed
        loss: loss value
    """

    # L1 distance
    dis_p_l1 = torch.norm(p - p_reconstructed, dim=-1)
    # weighted loss
    dis_p_dcd = 1 - torch.exp(-alpha * (dis_p_l1 ** 2))
    lh_p = 1 - dis_p_dcd
    confidence_normalized = confidence / (confidence.mean(dim=-1, keepdim=True) + 1e-6)
    loss_dis_p = (dis_p_dcd * confidence_normalized).mean()

    return lh_p, loss_dis_p


def loss_chamfer(p, q, R, t, s, confidence, alpha=50):
    """
    Compute the loss between s * (p @ R) + t and q using Chamfer distance.
    The loss is weighted by the confidence scores.
    
    Args:
        p: (B, N, 3) original points
        q: (B, N, 3) transformed points
        R: (B, 3, 3) rotation matrices
        t: (B, 3) translation vectors
        s: (B,) scale factors
    
    Returns:
        l1_distance: (B, N) L1 distance between transformed p and q
        loss: rigidity loss
    """
    confidence_normalized = confidence / (confidence.mean(dim=-1, keepdim=True) + 1e-6)  # normalize confidence
    #confidence_normalized = confidence_normalized.detach()  # stop gradient
    transformed_points = apply_transformation(p, R, t, s)
    rigidity_dis_l1 = chamfer_distance(transformed_points, q, norm=1, single_directional=True, point_reduction=None, batch_reduction=None)[0]  # (B,N)
    rigidity_dis_dcd = 1 - torch.exp(-alpha * (rigidity_dis_l1 ** 2))  # type: ignore
    lh = 1 - rigidity_dis_dcd
    loss_rigidity = (rigidity_dis_dcd * confidence_normalized).mean()
    return lh, loss_rigidity


def loss_stego(d1, d2, s1, s2, bias = 0.):
    """
    Stego loss to align DINO-based and STEGO-based attention matrices.
    
    Args:
        d_1: (B, N, D1) DINO features from point cloud p
        d_2: (B, M, D1) DINO features from point cloud q
        s_1: (B, N, D2) STEGO features from point cloud p
        s_2: (B, M, D2) STEGO features from point cloud q
        bias: bias term, bigger bias encourages less matching, smaller bias encourages more matching
    Returns:
        loss: stego loss
    """
    B = d1.shape[0]

    d1_normalized = F.normalize(d1, p=2, dim=-1)
    d2_normalized = F.normalize(d2, p=2, dim=-1)
    dino_m = torch.bmm(d1_normalized, d2_normalized.transpose(-2, -1))
    dino_m_mean = torch.mean(dino_m.reshape(B,-1), dim=-1)
    dino_m_normalized = dino_m - dino_m_mean.reshape(B,1,1)
    dino_m_biased = dino_m_normalized - bias

    s1_normalized = F.normalize(s1, p=2, dim=-1)
    s2_normalized = F.normalize(s2, p=2, dim=-1)
    stego_m = torch.bmm(s1_normalized, s2_normalized.transpose(-2, -1))
    stego_m_clamped = torch.clamp(stego_m, min=0.)

    loss = - dino_m_biased * stego_m_clamped
    theoretical_min = -(1.0 - bias) # avoid negative loss
    return loss.mean() - theoretical_min

def loss_semantic(geo_attn, sem_sim, confidence=None, alpha=2):
    """
    Geo-semantic consistency loss to align geometric attention with semantic similarity.
    
    Args:
        geo_attn: (B, N, M) geometric attention weights from p to q
        sem_sim: (B, N, M) semantic similarity matrix between p and q
        confidence: (B, N) confidence scores for points in p
    
    Returns:
        loss: geo-semantic consistency loss
    """

    # Calculate semantic distance
    diff = 1 - sem_sim
    #adactive_steepness = torch.tanh(diff / transition_point)
    #sem_dis = torch.exp(adactive_steepness * diff) - 1  # (B, N, M)
    sem_dis = 1 - torch.exp(-alpha * diff) 

    # Calculate row-wise weighted distance
    weighted_dis = (geo_attn * sem_dis).sum(dim=-1)  # (B, N)
    if confidence is not None:
        confidence_normalized = confidence / (confidence.mean(dim=-1, keepdim=True) + 1e-6)  # normalize confidence
        loss = (weighted_dis * confidence_normalized).mean()  # weight by confidence
    else:
        loss = weighted_dis.mean()
    
    likelyhood = 1 - weighted_dis
    
    return loss, likelyhood

def loss_reg_conf(confidence, target_mean=1., transition_point=0.2):
    """
    Regularization loss with log-exp combination for smooth adaptive steepness.
    
    Args:
        p_confidence: (B, N) confidence scores
        target_mean: target mean confidence value
        transition_point: point where loss transitions from gentle to steep
    
    Returns:
        loss: regularization loss
    """
    if target_mean >= 1.0:
        target_mean = 1-1e-3 # avoid target mean being 1.0
    batch_mean_confidence = confidence.mean(dim=-1)  # (B,)
    diff = target_mean - batch_mean_confidence
    diff_pos = torch.clamp(diff, min=0.0)
    
    # Adaptive steepness: gentle when diff is small, steep when diff is large
    # Use log(1 + x) for small values, exp(x) for large values
    adaptive_steepness = torch.tanh(diff_pos / transition_point)
    reg_loss = (torch.exp(adaptive_steepness * diff_pos) - 1.0).mean()

    return reg_loss

def loss_reg_conf_softplus(confidence, transition_point=0.2, temp=5):
    batch_mean_confidence = confidence.mean(dim=-1)  # (B,)
    loss = F.softplus(transition_point - batch_mean_confidence, beta=temp)
    return loss.mean()

def loss_rt(p1, pred_r, pred_t, gt_r, gt_t, c=None, alpha=50):
    """
    Rotation and translation loss.
    
    Args:
        p1: (B, N, 3) original points
        pred_r: (B, 3, 3) predicted rotation matrices
        pred_t: (B, 3) predicted translation vectors
        gt_r: (B, 3, 3) ground truth rotation matrices
        gt_t: (B, 3) ground truth translation vectors
        c: (B, N) confidence scores for points in p1
        alpha: dcd alpha
    Returns:
        loss: rotation and translation loss
    """
    pred_p1 = apply_transformation(p1, pred_r, pred_t)
    # Notice gt is camera pose
    gt_p1 = (p1 - gt_t.unsqueeze(1)) @ gt_r
    dis_l1 = torch.norm(pred_p1 - gt_p1, dim=-1)  # (B,N)
    dis_r_dcd = 1 - torch.exp(-alpha * (dis_l1 ** 2))
    if c is not None:
        c_normalized = c / (c.mean(dim=-1, keepdim=True) + 1e-6)
        loss = (dis_r_dcd * c_normalized).mean(1)
    else:
        loss = dis_r_dcd.mean(1)
    return loss

def compute_supervise_loss(
    end_points,
    pts1, pts2,
    pred_r, pred_t,
    gt_r, gt_t,
    alpha=50,
    loss_str="coarse_supervise",
    loss_weight=1,
    ):

    # calculate pose loss
    # loss_rt_12 = loss_rt(pts1, pred_r, pred_t, gt_r, gt_t, c1)
    # pred_inv_r, pred_inv_t, _ = inverse_transformation(pred_r, pred_t, torch.ones(B).to(pred_r.device))
    # gt_inv_r, gt_inv_t = gt_r.transpose(1, 2), (-gt_t.unsqueeze(1) @ gt_r).squeeze(1)
    # loss_rt_21 = loss_rt(pts2, pred_inv_r, pred_inv_t, gt_inv_r, gt_inv_t, c2)
    # loss_rt_ret = (loss_rt_12 + loss_rt_21) / 2
    loss_rt_ret = loss_rt(pts1, pred_r, pred_t, gt_r, gt_t, alpha=alpha)
    loss_pose = loss_rt_ret.mean()
    end_points[loss_str + "_pose_loss"] = loss_weight * loss_pose

    return end_points


def compute_conf_bce_loss(
        end_points,
        c1, c2,
        label1, label2,
        w_p=1., w_n=1.,
        loss_str="conf",
        loss_weight=1
        ):
    """
    Compute the confidence BCE loss and store in end_points.
    Args:
        end_points: dictionary to store loss values
        p: (B, N, 3) original points
        p_corr: (B, N, 3) corresponding points
        q: (B, N, 3) transformed points
        q_corr: (B, N, 3) corresponding points
        c1: (B, N) confidence scores for points in p
        c2: (B, N) confidence scores for points in q
        r: (B, 3, 3) rotation matrices
        t: (B, 3) translation vectors
        w_p, w_n: global class weights for pos/neg samples
        alpha: hyperparameter for the Gaussian RBF
        loss_str: prefix for loss key names
        loss_weight: weight for the loss
    Returns:
        end_points: updated dictionary with loss values
    """

    #loss_1= get_weighted_bce_loss(c1, label1, w_p, w_n).mean() 
    #loss_2= get_weighted_bce_loss(c2, label2, w_p, w_n).mean()
    loss_1= get_weighted_bce_loss_soft(c1, label1, w_p, w_n).mean()
    loss_2= get_weighted_bce_loss_soft(c2, label2, w_p, w_n).mean()
    loss = (loss_1 + loss_2) / 2
    end_points[f"{loss_str}_loss"] = loss * loss_weight
    return end_points


def get_corr_distance(p, r, t, p_corr, alpha=50):
    """
    Compute the distance between transformed points p and their correspondences p_corr.
    
    Args:
        p: (B, N, 3) original points
        r: (B, 3, 3) rotation matrices
        t: (B, 3) translation vectors
        p_corr: (B, N, 3) corresponding points
        alpha: hyperparameter for the Gaussian RBF
    Returns:
        dis_l1: (B, N) L1 distance between transformed p and p_corr
        rbf_label: (B, N) RBF-based soft labels (closer 1, farther 0)
    """
    p_transformed = apply_transformation(p, r, t)
    dis_l1 = torch.norm(p_transformed - p_corr, dim=-1)  # (B,N)
    rbf_label = torch.exp(-alpha * (dis_l1 ** 2))
    return dis_l1, rbf_label

def get_weighted_bce_loss_soft(prediction, gt, w_p=1., w_n=1., eps=1e-6):
    """
    prediction: (B, N)
    gt        : (B, N) soft label in [0,1]
    w_p, w_n  : global class weights for pos/neg samples
    """

    pred_prob = prediction.clamp(min=eps, max=1-eps)
    per_elem_loss = F.binary_cross_entropy(pred_prob, gt, reduction="none")

    # —— Soft class weights ——
    pos_ratio = gt.mean(dim=1, keepdim=True)
    neg_ratio = 1.0 - pos_ratio
    weights = gt * (neg_ratio * w_p) + (1.0 - gt) * (pos_ratio * w_n)

    w_class_loss = (weights * per_elem_loss).mean(dim=1)
    return w_class_loss

def get_weighted_bce_loss(prediction, gt, w_p=1., w_n=1., gamma=2.):
    loss = nn.BCELoss(reduction="none")

    class_loss = loss(prediction, gt)
    focal_weight = torch.where(gt >= 0.5, (1 - prediction) ** gamma, prediction ** gamma)
    class_loss = focal_weight * class_loss

    weights = torch.ones_like(gt)
    w_negative = gt.sum(1) / gt.size(1)
    w_positive = 1 - w_negative

    w_positive = w_positive[:, None].repeat(1, gt.shape[1])
    w_negative = w_negative[:, None].repeat(1, gt.shape[1])

    weights[gt >= 0.5] = w_positive[gt >= 0.5] * w_p
    weights[gt < 0.5] = w_negative[gt < 0.5] * w_n
    w_class_loss = (weights * class_loss).mean(1)

    #######################################
    # get classification precision and recall
    # predicted_labels = prediction.detach().cpu().round().numpy()
    # cls_precision, cls_recall, _, _ = precision_recall_fscore_support(gt.cpu().numpy(),predicted_labels, average='binary')

    return w_class_loss