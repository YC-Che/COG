import torch
import torch.nn as nn
import torch.nn.functional as F

from core.COG.model.transformer import GeometricTransformer
from core.COG.cogs.utils import sinkhorn_weight_prior, umeyama_bidirectional
from core.COG.cogs.loss import compute_cycle_loss, compute_chamfer_loss, compute_semantic_loss, compute_conf_bce_loss, compute_supervise_loss


class CoarsePointMatchingOneRef(nn.Module):
    def __init__(self, cfg, return_feat=False):
        super(CoarsePointMatchingOneRef, self).__init__()
        self.cfg = cfg
        self.return_feat = return_feat
        self.nblock = self.cfg.nblock

        self.in_proj = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.out_dim)
        self.conf_proj = nn.Linear(cfg.hidden_dim, 1)
        self.bg_token = nn.Parameter(torch.randn(1, 1, cfg.hidden_dim) * 0.02)

        self.transformers = []
        for _ in range(self.nblock):
            self.transformers.append(
                GeometricTransformer(
                    blocks=["self", "cross"],
                    d_model=cfg.hidden_dim,
                    num_heads=4,
                    dropout=None,
                    activation_fn="ReLU",
                    return_attention_scores=False,
                )
            )
        self.transformers = nn.ModuleList(self.transformers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, p1, f1, geo1, p2, f2, geo2, radius, end_points):
        B = f1.size(0)

        # semantic similarity
        f_sim = torch.matmul(f1, f2.transpose(-2,-1))

        # Geometric features
        f_1 = self.in_proj(f1)
        f_1 = torch.cat([self.bg_token.repeat(B, 1, 1), f_1], dim=1)  # adding bg
        f_2 = self.in_proj(f2)
        f_2 = torch.cat([self.bg_token.repeat(B, 1, 1), f_2], dim=1)  # adding bg
        for idx in range(self.nblock):
            f_1, f_2 = self.transformers[idx](f_1, geo1, f_2, geo2)
        g1 = self.out_proj(f_1)[:,1:] # remove bg token
        g1 = F.normalize(g1, p=2, dim=-1)
        g2 = self.out_proj(f_2)[:,1:] # remove bg token
        g2 = F.normalize(g2, p=2, dim=-1)
        g_sim = torch.matmul(g1, g2.transpose(-2, -1)) # (B, N1, N2)

        # confidence
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
        end_points["init_R"] = r
        end_points["init_t"] = t

        if self.training:
            # Supervise loss for coarse matching
            if self.cfg.loss_supervise_w > 0:
                compute_supervise_loss(
                    end_points,
                    p1, p2,
                    r, t,
                    end_points["rotation_label"],
                    end_points["translation_label"] / (radius.reshape(-1, 1) + 1e-6),
                    alpha=self.cfg.loss_supervise_alpha,
                    loss_str="coarse_supervise",
                    loss_weight=self.cfg.loss_supervise_w
                )

            # Cycle consistency loss
            end_points, lh_cycl1, lh_cycl2 = compute_cycle_loss(
                end_points,
                p1, p1_recon, c1,
                p2, p2_recon, c2,
                alpha=self.cfg.loss_cycle_alpha,
                loss_str="coarse_cycle",
                loss_weight=self.cfg.loss_cycle_w
            )

            # Chamfer loss
            end_points, lh_chamfer1, lh_chamfer2 = compute_chamfer_loss(
                end_points,
                p1, p2,
                r, t, s,
                c1, c2,
                alpha=self.cfg.loss_chamfer_alpha,
                loss_str="coarse_chamfer",
                loss_weight=self.cfg.loss_chamfer_w
            )

            # Semantic consistency loss
            end_points, lh_sem1, lh_sem2 = compute_semantic_loss(
                end_points,
                operator_12, operator_21,
                c1, c2,
                f_sim,
                alpha=self.cfg.loss_semantic_alpha,
                loss_str="coarse_semantic",
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
            compute_conf_bce_loss(
                end_points,
                c1, c2,
                lh_label_1, lh_label_2,
                w_p=self.cfg.loss_conf_w_p,
                w_n=self.cfg.loss_conf_w_n,
                loss_str="coarse_conf",
                loss_weight=self.cfg.loss_conf_w
            )

        if self.return_feat:
            return end_points, g1, g2
        else:
            return end_points
