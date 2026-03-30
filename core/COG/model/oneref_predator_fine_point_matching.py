import torch
from torch.cuda.amp import autocast
import torch.nn as nn
import torch.nn.functional as F

from core.COG.model.transformer import SparseToDenseTransformer
from core.COG.model.pointnet2.pointnet2_utils import QueryAndGroup, QueryAndLRFGroup
from core.COG.model.pointnet2.pytorch_utils import SharedMLP, Conv1d
from core.COG.cogs.utils import sinkhorn_weight_prior, umeyama_bidirectional, apply_transformation, inverse_transformation
from core.COG.cogs.loss import compute_cycle_loss, compute_chamfer_loss, compute_semantic_loss, compute_conf_bce_loss, compute_supervise_loss

class FinePointMatchingOneRef(nn.Module):
    def __init__(self, cfg, return_feat=False):
        super(FinePointMatchingOneRef, self).__init__()
        self.cfg = cfg
        self.return_feat = return_feat
        self.nblock = self.cfg.nblock

        self.in_proj = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.out_dim)
        self.conf_proj = nn.Linear(cfg.hidden_dim, 1)

        self.bg_token = nn.Parameter(torch.randn(1, 1, cfg.hidden_dim) * 0.02)
        self.PE = PositionalEncoding(
            cfg.hidden_dim,
            r1=cfg.pe_radius1,
            r2=cfg.pe_radius2,
            nsample1=cfg.get("nsample1", 32),
            nsample2=cfg.get("nsample2", 64),
            use_lrf=cfg.use_lrf,
            use_xyz=cfg.use_xyz,
            use_feature=cfg.get("use_feature", False),
        )

        self.transformers = []
        for _ in range(self.nblock):
            self.transformers.append(
                SparseToDenseTransformer(
                    cfg.hidden_dim,
                    num_heads=4,
                    sparse_blocks=["self", "cross"],
                    dropout=None,
                    activation_fn="ReLU",
                    focusing_factor=cfg.focusing_factor,
                    with_bg_token=True,
                    replace_bg_token=True,
                )
            )
        self.transformers = nn.ModuleList(self.transformers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, p1, f1, geo1, fps_idx1, p2, f2, geo2, fps_idx2, radius, end_points, refine_t=0):
        """
        p1 (B, n1, 3): tgt, m, observed
        p2 (B, n2, 3): src, o, rendered
        """
        B = p1.size(0)
        if "init_R" in end_points and "init_t" in end_points:
            # Detach to prevent gradient computation issues when reusing tensors from previous network
            # This stops gradient flow but preserves the transformation for initialization
            init_R = end_points["init_R"].detach()
            init_t = end_points["init_t"].detach()
            p1_ = apply_transformation(p1, init_R, init_t)
            p1_ = p1_ - p2.mean(dim=1, keepdim=True)
        else:
            p1_ = p1 - p1.mean(dim=1, keepdim=True)
        p2_ = p2 - p2.mean(dim=1, keepdim=True)

        # Semantic similarity
        f_sim = torch.matmul(f1, f2.transpose(-2,-1))

        # Geometric features
        f_1 = self.in_proj(f1) + self.PE(p1_)
        f_1 = torch.cat([self.bg_token.repeat(B, 1, 1), f_1], dim=1)  # adding bg
        f_2 = self.in_proj(f2) + self.PE(p2_)
        f_2 = torch.cat([self.bg_token.repeat(B, 1, 1), f_2], dim=1)  # adding bg
        for idx in range(self.nblock):
            f_1, f_2 = self.transformers[idx](f_1, geo1, fps_idx1, f_2, geo2, fps_idx2)
        g1 = self.out_proj(f_1)[:,1:] # remove bg token
        g1 = F.normalize(g1, p=2, dim=-1)
        g2 = self.out_proj(f_2)[:,1:] # remove bg token
        g2 = F.normalize(g2, p=2, dim=-1)
        g_sim = torch.matmul(g1, g2.transpose(-2, -1)) # (B, N1, N2)

        # confidence scores
        c1 = self.conf_proj(f_1).squeeze(-1)[:, 1:]
        c1 = F.sigmoid(c1).clamp(min=1e-6, max=1-1e-6)
        c2 = self.conf_proj(f_2).squeeze(-1)[:, 1:]
        c2 = F.sigmoid(c2).clamp(min=1e-6, max=1-1e-6)

        # Optimal transport with Sinkhorn
        trans_plan = sinkhorn_weight_prior(
            g_sim, c1, c2,
            num_iters=self.cfg.sinkhorn_iter, temperature=self.cfg.temp, normalize=True, use_checkpoint=self.training,
            prior=f_sim + 1, prior_weight=self.cfg.prior_weight
            )
        operator_12 = F.normalize(trans_plan, p=1, dim=-1, eps=1e-6) # (B, N1, N2)
        operator_21 = F.normalize(trans_plan.transpose(-2, -1), p=1, dim=-1, eps=1e-6) # (B, N2, N1)
        p1_corr = torch.matmul(operator_21, p1) # (B, N2, 3) corresponding points in p1 space for p2
        p2_corr = torch.matmul(operator_12, p2) # (B, N1, 3) corresponding points in p2 space for p1
        p1_recon = torch.matmul(operator_12, p1_corr) # (B, N1, 3) reconstructed points in p1 space for p1
        p2_recon = torch.matmul(operator_21, p2_corr) # (B, N2, 3) reconstructed points in p2 space for p2

        # Umeyama SVD for rigid transformation estimation
        r,t,s = umeyama_bidirectional(p1, p1_corr, p2, p2_corr, c1, c2, estimate_scale=False)

        if self.training:
            # Supervise loss for rotation and translation
            if self.cfg.loss_supervise_w > 0:
                compute_supervise_loss(
                    end_points,
                    p1, p2,
                    r, t,
                    end_points["rotation_label"],
                    end_points["translation_label"] / (radius.reshape(-1, 1) + 1e-6),
                    alpha=self.cfg.loss_supervise_alpha,
                    loss_str="fine_supervise",
                    loss_weight=self.cfg.loss_supervise_w
                )

            # Cycle consistency loss
            end_points, lh_cycl1, lh_cycl2 = compute_cycle_loss(
                end_points,
                p1, p1_recon, c1,
                p2, p2_recon, c2,
                alpha=self.cfg.loss_cycle_alpha,
                loss_str="fine_cycle",
                loss_weight=self.cfg.loss_cycle_w
            )

            # Chamfer loss
            end_points, lh_chamfer1, lh_chamfer2 = compute_chamfer_loss(
                end_points,
                p1, p2,
                r, t, s,
                c1, c2,
                alpha=self.cfg.loss_chamfer_alpha,
                loss_str="fine_chamfer",
                loss_weight=self.cfg.loss_chamfer_w
            )

            # Semantic consistency loss
            end_points, lh_sem1, lh_sem2 = compute_semantic_loss(
                end_points,
                operator_12, operator_21,
                c1, c2,
                f_sim,
                alpha=self.cfg.loss_semantic_alpha,
                loss_str="fine_semantic",
                loss_weight=self.cfg.loss_semantic_w
            )

            # pseudo label for confidence learning
            lh_sem1_max = lh_sem1.max(dim=-1, keepdim=True).values
            lh_sem2_max = lh_sem2.max(dim=-1, keepdim=True).values
            lh_sem1 = lh_sem1 + (1 - lh_sem1_max) # ensure max is 1
            lh_sem2 = lh_sem2 + (1 - lh_sem2_max) # ensure max is 1
            cycl_w, chamfer_w, sem_w = self.cfg.loss_conf_cycle_w, self.cfg.loss_conf_chamfer_w, self.cfg.loss_conf_semantic_w
            lh_label_1 = lh_cycl1.pow(cycl_w) * lh_chamfer1.pow(chamfer_w) * lh_sem1.pow(sem_w)
            lh_label_2 = lh_cycl2.pow(cycl_w) * lh_chamfer2.pow(chamfer_w) * lh_sem2.pow(sem_w)
            lh_label_1 = lh_label_1.detach()
            lh_label_2 = lh_label_2.detach()

            # Confidence loss
            end_points = compute_conf_bce_loss(
                end_points,
                c1, c2,
                lh_label_1, lh_label_2,
                w_p=self.cfg.loss_conf_w_p,
                w_n=self.cfg.loss_conf_w_n,
                loss_str="fine_conf",
                loss_weight=self.cfg.loss_conf_w
            )

            return end_points

        else:
            # During inference, iterative refinement for a few steps
            if refine_t < self.cfg.refine_t:
                end_points["init_R"] = r
                end_points["init_t"] = t
                return self.forward(p1, f1, geo1, fps_idx1, p2, f2, geo2, fps_idx2, radius, end_points, refine_t=refine_t+1)
            else:
                eval_r = r
                eval_t = (- t.unsqueeze(1) @ r.transpose(1,2)).squeeze(1)
                inv_r, inv_t, _ = inverse_transformation(r,t,torch.ones_like(s))
                score = compute_score(p2, p1, inv_r, inv_t, dis_thres=0.05) # compute score in p2 space for better evaluation of fine matching
                end_points["pred_R"] = eval_r
                end_points["pred_t"] = eval_t * (radius.reshape(-1, 1) + 1e-6)
                end_points["pred_pose_score"] = score
                return end_points


class PositionalEncoding(nn.Module):
    def __init__(
        self, out_dim, r1=0.1, r2=0.2, nsample1=32, nsample2=64, use_lrf=True, use_xyz=False, use_feature=False, bn=True
    ):
        super(PositionalEncoding, self).__init__()
        if use_lrf:
            self.group1 = QueryAndLRFGroup(r1, nsample1, use_xyz=use_xyz, use_feature=use_feature)
            self.group2 = QueryAndLRFGroup(r2, nsample2, use_xyz=use_xyz, use_feature=use_feature)
        else:
            self.group1 = QueryAndGroup(r1, nsample1, use_xyz=use_xyz)
            self.group2 = QueryAndGroup(r2, nsample2, use_xyz=use_xyz)
        input_dim = 3
        if use_xyz:
            input_dim += 3
        if use_feature:
            input_dim += 3

        self.mlp1 = SharedMLP([input_dim, 32, 64, 128], bn=bn)
        self.mlp2 = SharedMLP([input_dim, 32, 64, 128], bn=bn)
        self.mlp3 = Conv1d(256, out_dim, 1, activation=None, bn=None)

    def forward(self, pts1, pts2=None):
        if pts2 is None:
            pts2 = pts1

        pts1 = pts1.to(dtype=torch.float32)
        pts2 = pts2.to(dtype=torch.float32)
        with autocast(enabled=False):
            # scale1
            feat1 = self.group1(pts1.contiguous(), pts2.contiguous(), pts1.transpose(1, 2).contiguous())
            feat1 = self.mlp1(feat1)
            feat1 = F.max_pool2d(feat1, kernel_size=[1, feat1.size(3)])

            # scale2
            feat2 = self.group2(pts1.contiguous(), pts2.contiguous(), pts1.transpose(1, 2).contiguous())
            feat2 = self.mlp2(feat2)
            feat2 = F.max_pool2d(feat2, kernel_size=[1, feat2.size(3)])

            feat = torch.cat([feat1, feat2], dim=1).squeeze(-1)
            feat = self.mlp3(feat).transpose(1, 2)
            return feat


def compute_score(pts1, pts2, pred_R, pred_t, dis_thres=0.2):
    pts1_trans = apply_transformation(pts1, pred_R, pred_t)
    dists = torch.cdist(pts1_trans, pts2, p=2).min(dim=-1).values
    inlier_mask = (dists < dis_thres).float()
    scores = inlier_mask.sum(dim=1) / pts1.size(1)
    return scores
