"""
═══════════════════════════════════════════════════════════════════════════════
 UNET V2 — IMPROVED PARAMETRIC FIELD SURROGATE
 Implements ALL Part 1 improvements over the baseline SSIM model:

 [1] U-Net skip connections    → sharp quench front, fine spatial detail
 [2] Relative time t/t_end    → proper temporal evolution
 [3] SSIM win_size=5 + TV loss → eliminates 9px block artefacts
 [4] Front-weighted loss        → model focuses on high-|∇T| regions
 [5] Multi-scale FiLM           → scalar params modulate every decoder level

 Architecture:
   ScalarEncoder (MLP)  : [9 params + t_rel + t_end_norm] → latent(256)
   SpatialEncoder (CNN) : POR(1,101,89) → skip features at 4 scales
   FiLM fusion          : multiplicative + additive at each decoder level
   Decoder (U-Net)      : 3× ConvTranspose + skip concat → (101,89)
   Output               : Sigmoid → normalised T ∈ [0,1]

 Loss:
   L = 0.50·relMSE + 0.15·SSIM(win=5) + 0.20·gradient + 0.10·TV + 0.05·front

 n_params = 11  (9 physical + t_rel + t_end_norm)
 Dataset  : ~/energy_PINN_training/2D_Txzt_model/dataset/
 Results  : ~/energy_PINN_training/2D_Txzt_model/results/UNET_V2/
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import time
import logging
import datetime
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from pytorch_msssim import SSIM
except ImportError:
    raise ImportError("Run: pip install pytorch-msssim")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
class Config:
    # ── paths ─────────────────────────────────────────────────────────────────
    dataset_dir = '/home/jasminjthompson/energy_PINN_training/2D_Txzt_model/dataset'
    results_dir = '/home/jasminjthompson/energy_PINN_training/2D_Txzt_model/results/UNET_V2'

    # ── data ──────────────────────────────────────────────────────────────────
    # n_params = 11: 9 physical + t_rel (t/t_end) + t_end_norm (t_end/T_SIM_MAX)
    # This is the KEY change from baseline (10 params) — relative time
    n_params   = 11
    NZ, NX     = 101, 89
    T_MIN      = 376.0
    T_MAX      = 2700.1
    T_SIM_MAX  = 7200.0

    # ── ensemble ──────────────────────────────────────────────────────────────
    n_members  = 3
    seeds      = [0, 1, 2]

    # ── training ──────────────────────────────────────────────────────────────
    batch_size   = 64
    epochs       = 150
    lr           = 1e-3
    weight_decay = 1e-4
    grad_clip    = 1.0
    latent_dim   = 256
    num_workers  = 4
    patience     = 25   # longer patience than baseline

    # ── loss weights (sum to 1.0) ──────────────────────────────────────────────
    # [1] relative MSE  — robust to low-contrast frames
    # [2] SSIM win=5    — structural fidelity, no 9px tiling artefact
    # [3] gradient      — quench front sharpness
    # [4] TV            — suppresses block artefacts
    # [5] front         — extra weight on high-|∇T| cells
    w_mse   = 0.50
    w_ssim  = 0.15
    w_grad  = 0.20
    w_tv    = 0.10
    w_front = 0.05
    front_weight_scale = 5.0   # max weight at front vs background

    # SSIM window — 5×5 avoids 9px tiling seen with default 11×11
    ssim_win_size = 5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════
def setup_logging(results_dir):
    os.makedirs(results_dir, exist_ok=True)
    fmt = logging.Formatter('%(asctime)s  %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    log = logging.getLogger('unet_v2')
    log.setLevel(logging.INFO)
    if not log.handlers:
        fh = logging.FileHandler(os.path.join(results_dir, 'training.log'))
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        log.addHandler(fh)
        log.addHandler(sh)
    return log


def write_progress(cfg, member_id, epoch, tr, va,
                   tr_mse, va_mse, tr_ssim, va_ssim,
                   tr_grad, va_grad, tr_tv, va_tv, tr_front, va_front,
                   best_val, best_ep, elapsed, no_imp):
    eta = (elapsed / max(epoch, 1)) * (cfg.epochs - epoch)
    with open(os.path.join(cfg.results_dir,
              f'progress_member{member_id}.json'), 'w') as fh:
        json.dump({
            'member': member_id, 'epoch': epoch,
            'total_epochs': cfg.epochs,
            'train_loss': round(tr, 8),   'val_loss':  round(va, 8),
            'train_mse':  round(tr_mse, 8), 'val_mse':   round(va_mse, 8),
            'train_ssim': round(tr_ssim, 8),'val_ssim':  round(va_ssim, 8),
            'train_grad': round(tr_grad, 8),'val_grad':  round(va_grad, 8),
            'train_tv':   round(tr_tv, 8),  'val_tv':    round(va_tv, 8),
            'train_front':round(tr_front,8),'val_front': round(va_front,8),
            'best_val': round(best_val, 8), 'best_epoch': best_ep,
            'no_improve': no_imp, 'patience': cfg.patience,
            'elapsed_min': round(elapsed / 60, 1),
            'eta_min':     round(eta / 60, 1),
            'updated': datetime.datetime.now().isoformat(),
        }, fh, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# Loads X_train.npy etc. with mmap — no full copy in RAM.
# X has shape (N, 11):  9 params + t_rel + t_end_norm
# Y has shape (N, 101, 89): normalised temperature field
# M has shape (N, 101, 89): binary bed mask
# POR has shape (N, 1, 101, 89): normalised porosity
# ═══════════════════════════════════════════════════════════════════════════════
class FieldDataset(Dataset):
    def __init__(self, dataset_dir, split):
        def load(name):
            return np.load(
                os.path.join(dataset_dir, f'{name}_{split}.npy'),
                mmap_mode='r')
        self.X   = load('X')
        self.Y   = load('Y')
        self.M   = load('M')
        self.POR = load('POR')
        assert len(self.X) == len(self.Y) == len(self.M) == len(self.POR), \
            f"Array length mismatch for split '{split}'"

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        a = lambda x: torch.from_numpy(np.asarray(x, dtype=np.float32).copy())
        return a(self.X[i]), a(self.POR[i]), a(self.Y[i]), a(self.M[i])


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

class ScalarEncoder(nn.Module):
    """MLP: 11 scalar inputs → latent(256).
    Input: [9 params + t_rel + t_end_norm]
    t_rel = t/t_end  ∈ [0,1]: relative position in simulation (KEY fix)
    t_end_norm = t_end/7200 ∈ [0,1]: how long the sim runs (encodes outcome)
    """
    def __init__(self, n_params, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_params, 128),     nn.LayerNorm(128),      nn.GELU(),
            nn.Linear(128,      256),     nn.LayerNorm(256),      nn.GELU(),
            nn.Linear(256, latent_dim),   nn.LayerNorm(latent_dim), nn.GELU(),
        )
    def forward(self, x):
        return self.net(x)


class FiLMBlock(nn.Module):
    """Full FiLM: multiplicative scale + additive shift.
    More expressive than the purely additive version in the baseline.
    Reference: Perez et al. AAAI 2018.
    """
    def __init__(self, spatial_ch, param_dim):
        super().__init__()
        self.gamma = nn.Linear(param_dim, spatial_ch)
        self.beta  = nn.Linear(param_dim, spatial_ch)

    def forward(self, spatial, params):
        """
        spatial: (B, C, H, W)
        params:  (B, param_dim)
        """
        g = self.gamma(params)[:, :, None, None]  # (B, C, 1, 1)
        b = self.beta(params)[:, :, None, None]
        return g * spatial + b


class UNetSurrogate(nn.Module):
    """
    U-Net with multi-scale FiLM conditioning.
    Decoder uses bilinear upsample + Conv2d instead of ConvTranspose2d
    to handle odd spatial dimensions (101, 89) without size mismatches.
    Skip connections use F.interpolate to match encoder sizes exactly.
    """
    def __init__(self, cfg):
        super().__init__()
        self.nz, self.nx = cfg.NZ, cfg.NX
        ld = cfg.latent_dim

        # scalar encoder
        self.scalar_enc = ScalarEncoder(cfg.n_params, ld)

        # spatial encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1,  16, 3, padding=1),           nn.BatchNorm2d(16),  nn.GELU())
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32),  nn.GELU())
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64),  nn.GELU())
        self.enc4 = nn.Sequential(
            nn.Conv2d(64,128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU())

        self.film_bottleneck = FiLMBlock(128, ld)

        # decoder: upsample + conv (NOT ConvTranspose2d)
        # input channels = upsampled + skip
        self.dec3_conv = nn.Sequential(
            nn.Conv2d(128+64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU())
        self.film3 = FiLMBlock(64, ld)

        self.dec2_conv = nn.Sequential(
            nn.Conv2d(64+32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU())
        self.film2 = FiLMBlock(32, ld)

        self.dec1_conv = nn.Sequential(
            nn.Conv2d(32+16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.GELU())
        self.film1 = FiLMBlock(16, ld)

        self.out_conv = nn.Sequential(
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid())

    def forward(self, x, por):
        s  = self.scalar_enc(x)        # (B, latent_dim)

        # encoder
        e1 = self.enc1(por)            # (B,  16, 101,  89)
        e2 = self.enc2(e1)             # (B,  32,  51,  45)
        e3 = self.enc3(e2)             # (B,  64,  26,  23)
        e4 = self.enc4(e3)             # (B, 128,  13,  12)
        e4 = self.film_bottleneck(e4, s)

        # decoder: upsample to EXACT encoder size then cat skip
        u3 = F.interpolate(e4, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3_conv(torch.cat([u3, e3], dim=1))   # (B,  64, 26, 23)
        d3 = self.film3(d3, s)

        u2 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2_conv(torch.cat([u2, e2], dim=1))   # (B,  32, 51, 45)
        d2 = self.film2(d2, s)

        u1 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1_conv(torch.cat([u1, e1], dim=1))   # (B,  16, 101, 89)
        d1 = self.film1(d1, s)

        out = self.out_conv(d1)        # (B,   1, 101,  89)
        return out.squeeze(1)          # (B, 101,  89) — exact size, no resize
# ═══════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

# SSIM module — win_size=5 avoids the 9px block artefact from default win=11
_ssim_module = None
def get_ssim(device, win_size):
    global _ssim_module
    if _ssim_module is None:
        _ssim_module = SSIM(
            data_range=1.0, size_average=True,
            win_size=win_size, win_sigma=0.5, channel=1
        ).to(device)
    return _ssim_module


def relative_mse(pred, target, mask, epsilon=0.01):
    """MSE normalised by per-sample field std.
    Prevents low-contrast frames from dominating training.
    Reference: Pathak et al. FourCastNet (arXiv 2022).
    """
    n   = mask.sum(dim=(-2,-1), keepdim=True).clamp(min=1)
    mu  = (target * mask).sum(dim=(-2,-1), keepdim=True) / n
    var = ((target - mu)**2 * mask).sum(dim=(-2,-1), keepdim=True) / n
    std = (var + epsilon).sqrt()
    return (((pred - target) / std)**2 * mask).sum() / (mask.sum() + 1e-8)


def ssim_loss(pred, target, mask, cfg):
    """1 - SSIM on masked field. win_size=5 eliminates 9px block tiling."""
    m  = mask.unsqueeze(1)
    p  = pred.unsqueeze(1)   * m
    t  = target.unsqueeze(1) * m
    return 1.0 - get_ssim(pred.device, cfg.ssim_win_size)(p, t)


def gradient_loss(pred, target, mask):
    """Penalises spatial gradient errors — sharpens quench front.
    Reference: Li & Snavely MegaDepth CVPR 2018.
    """
    dy_p = pred[:, 1:, :]   - pred[:, :-1, :]
    dy_t = target[:, 1:, :] - target[:, :-1, :]
    dx_p = pred[:, :, 1:]   - pred[:, :, :-1]
    dx_t = target[:, :, 1:] - target[:, :, :-1]
    m_dy = mask[:, 1:, :] * mask[:, :-1, :]
    m_dx = mask[:, :, 1:] * mask[:, :, :-1]
    ly = ((dy_p - dy_t)**2 * m_dy).sum() / (m_dy.sum() + 1e-8)
    lx = ((dx_p - dx_t)**2 * m_dx).sum() / (m_dx.sum() + 1e-8)
    return ly + lx


def tv_loss(pred, mask):
    """Total variation regularisation — suppresses block artefacts.
    Pixel-level smoothness penalty, no window tiling → no periodic blocks.
    """
    dy = (pred[:, 1:, :] - pred[:, :-1, :])**2
    dx = (pred[:, :, 1:] - pred[:, :, :-1])**2
    m_dy = mask[:, 1:, :] * mask[:, :-1, :]
    m_dx = mask[:, :, 1:] * mask[:, :, :-1]
    return ((dy*m_dy).sum()/(m_dy.sum()+1e-8) +
            (dx*m_dx).sum()/(m_dx.sum()+1e-8))


def front_weighted_loss(pred, target, mask, cfg):
    """Up-weights loss at the quench front (high |∇T| cells).
    Detects front from target gradient magnitude → soft spatial weight map.
    Reference: Chen et al. boundary loss CVPR 2019.
    """
    # gradient magnitude of target field
    dy = (target[:, 1:, :] - target[:, :-1, :]).abs()
    dx = (target[:, :, 1:] - target[:, :, :-1]).abs()
    grad_mag = torch.zeros_like(target)
    grad_mag[:, :-1, :] += dy
    grad_mag[:, 1:,  :] += dy
    grad_mag[:, :, :-1] += dx
    grad_mag[:, :,  1:] += dx
    # normalise per sample
    gmax = grad_mag.flatten(1).max(dim=1)[0][:, None, None].clamp(min=1e-6)
    grad_norm = grad_mag / gmax   # ∈ [0, 1]
    # weight: 1 in background, cfg.front_weight_scale at front
    w = 1.0 + (cfg.front_weight_scale - 1.0) * grad_norm
    err = (pred - target)**2 * mask * w
    return err.sum() / ((mask * w).sum() + 1e-8)


def combined_loss(pred, target, mask, cfg):
    """Weighted combination of all 5 loss terms."""
    l_mse   = relative_mse(pred, target, mask)
    l_ssim  = ssim_loss(pred, target, mask, cfg)
    l_grad  = gradient_loss(pred, target, mask)
    l_tv    = tv_loss(pred, mask)
    l_front = front_weighted_loss(pred, target, mask, cfg)
    total = (cfg.w_mse   * l_mse  +
             cfg.w_ssim  * l_ssim +
             cfg.w_grad  * l_grad +
             cfg.w_tv    * l_tv   +
             cfg.w_front * l_front)
    return total, l_mse, l_ssim, l_grad, l_tv, l_front


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def save_checkpoint(model, optimizer, epoch, val_loss, losses_dict,
                    seed, member_id, cfg, path):
    torch.save({
        'epoch'       : epoch,
        'model_state' : model.state_dict(),
        'optim_state' : optimizer.state_dict(),
        'val_loss'    : val_loss,
        'losses'      : losses_dict,
        'seed'        : seed,
        'member_id'   : member_id,
        'cfg': {
            'NZ': cfg.NZ, 'NX': cfg.NX,
            'n_params': cfg.n_params,
            'latent_dim': cfg.latent_dim,
            'T_MIN': cfg.T_MIN, 'T_MAX': cfg.T_MAX,
            'T_SIM_MAX': cfg.T_SIM_MAX,
        },
    }, path)


def save_onnx(model, cfg, path):
    """Export to ONNX for cross-platform loading."""
    model.eval()
    dummy_X   = torch.zeros(1, cfg.n_params,  device=cfg.device)
    dummy_POR = torch.zeros(1, 1, cfg.NZ, cfg.NX, device=cfg.device)
    torch.onnx.export(
        model, (dummy_X, dummy_POR), path,
        input_names=['X', 'POR'], output_names=['T_field'],
        dynamic_axes={'X': {0:'batch'}, 'POR': {0:'batch'}, 'T_field': {0:'batch'}},
        opset_version=17,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN ONE MEMBER
# ═══════════════════════════════════════════════════════════════════════════════
def train_member(member_id, seed, cfg, train_dl, val_dl):
    log = logging.getLogger('unet_v2')
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model     = UNetSurrogate(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs)

    n_p = sum(p.numel() for p in model.parameters())
    log.info(f"{'='*60}")
    log.info(f"MEMBER {member_id}  seed={seed}  params={n_p:,}")
    log.info(f"{'='*60}")

    hist = {k: [] for k in ['train_loss','val_loss',
                             'train_mse','val_mse',
                             'train_ssim','val_ssim',
                             'train_grad','val_grad',
                             'train_tv','val_tv',
                             'train_front','val_front','lr']}
    best_val   = np.inf
    best_ep    = 0
    no_imp     = 0
    ep_times   = []

    ckpt_path  = os.path.join(cfg.results_dir, f'model_member{member_id}.pt')
    final_path = os.path.join(cfg.results_dir, f'model_member{member_id}_final.pt')
    onnx_path  = os.path.join(cfg.results_dir, f'model_member{member_id}.onnx')
    hist_path  = os.path.join(cfg.results_dir, f'history_member{member_id}.json')
    prog_path  = os.path.join(cfg.results_dir, f'progress_member{member_id}.json')

    wall_start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        # ── train ─────────────────────────────────────────────────────────────
        model.train()
        tr = tr_mse = tr_ssim = tr_grad = tr_tv = tr_front = 0.0
        for X, POR, Y, M in train_dl:
            X, POR, Y, M = [t.to(cfg.device) for t in (X, POR, Y, M)]
            pred = model(X, POR)
            loss, lm, ls, lg, ltv, lf = combined_loss(pred, Y, M, cfg)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            n = len(X)
            tr     += loss.item()
            tr_mse += lm.item(); tr_ssim += ls.item()
            tr_grad+= lg.item(); tr_tv   += ltv.item(); tr_front += lf.item()

        nb = len(train_dl)
        tr/=nb; tr_mse/=nb; tr_ssim/=nb; tr_grad/=nb; tr_tv/=nb; tr_front/=nb

        # ── validate ──────────────────────────────────────────────────────────
        model.eval()
        va = va_mse = va_ssim = va_grad = va_tv = va_front = 0.0
        with torch.no_grad():
            for X, POR, Y, M in val_dl:
                X, POR, Y, M = [t.to(cfg.device) for t in (X, POR, Y, M)]
                pred = model(X, POR)
                loss, lm, ls, lg, ltv, lf = combined_loss(pred, Y, M, cfg)
                va     += loss.item()
                va_mse += lm.item(); va_ssim += ls.item()
                va_grad+= lg.item(); va_tv   += ltv.item(); va_front += lf.item()

        nb = len(val_dl)
        va/=nb; va_mse/=nb; va_ssim/=nb; va_grad/=nb; va_tv/=nb; va_front/=nb

        lr_now   = scheduler.get_last_lr()[0]
        scheduler.step()
        ep_t     = time.time() - t0
        elapsed  = time.time() - wall_start
        eta      = (elapsed / epoch) * (cfg.epochs - epoch)
        ep_times.append(ep_t)

        for k, v in zip(
            ['train_loss','val_loss','train_mse','val_mse',
             'train_ssim','val_ssim','train_grad','val_grad',
             'train_tv','val_tv','train_front','val_front','lr'],
            [tr,va,tr_mse,va_mse,tr_ssim,va_ssim,
             tr_grad,va_grad,tr_tv,va_tv,tr_front,va_front,lr_now]):
            hist[k].append(v)

        # ── checkpoint ────────────────────────────────────────────────────────
        if va < best_val:
            best_val = va; best_ep = epoch; no_imp = 0
            save_checkpoint(model, optimizer, epoch, va,
                            {'mse':va_mse,'ssim':va_ssim,'grad':va_grad,
                             'tv':va_tv,'front':va_front},
                            seed, member_id, cfg, ckpt_path)
        else:
            no_imp += 1

        # ── log ───────────────────────────────────────────────────────────────
        log.info(
            f"M{member_id} ep={epoch:3d}/{cfg.epochs}  "
            f"tr={tr:.5f}(mse={tr_mse:.4f} ssim={tr_ssim:.4f} "
            f"grad={tr_grad:.4f} tv={tr_tv:.4f} front={tr_front:.4f})  "
            f"va={va:.5f}(mse={va_mse:.4f} ssim={va_ssim:.4f} "
            f"grad={va_grad:.4f} tv={va_tv:.4f} front={va_front:.4f})  "
            f"best={best_val:.5f}@ep{best_ep}  "
            f"lr={lr_now:.2e}  ni={no_imp}/{cfg.patience}  "
            f"ep={ep_t:.1f}s  eta={eta/60:.1f}min"
        )

        write_progress(cfg, member_id, epoch, tr, va,
                       tr_mse,va_mse,tr_ssim,va_ssim,
                       tr_grad,va_grad,tr_tv,va_tv,tr_front,va_front,
                       best_val,best_ep,elapsed,no_imp)
        with open(hist_path, 'w') as fh:
            json.dump(hist, fh)

        if no_imp >= cfg.patience:
            log.info(f"M{member_id} early stop @ ep{epoch}")
            break

    # ── save final + ONNX ─────────────────────────────────────────────────────
    save_checkpoint(model, optimizer, epoch, va,
                    {'mse':va_mse,'ssim':va_ssim,'grad':va_grad,
                     'tv':va_tv,'front':va_front},
                    seed, member_id, cfg, final_path)

    best_ckpt = torch.load(ckpt_path, map_location=cfg.device,
                           weights_only=False)
    model.load_state_dict(best_ckpt['model_state'])
    save_onnx(model, cfg, onnx_path)

    wall_total = time.time() - wall_start
    summary = {
        'member_id'        : member_id,
        'seed'             : seed,
        'epochs_trained'   : epoch,
        'best_val_loss'    : best_val,
        'best_epoch'       : best_ep,
        'final_val_mse'    : va_mse,
        'final_val_ssim'   : va_ssim,
        'wall_time_min'    : round(wall_total/60, 2),
        'mean_epoch_time_s': round(float(np.mean(ep_times)), 2),
        'files': {
            'best_checkpoint': ckpt_path,
            'final_weights'  : final_path,
            'onnx'           : onnx_path,
            'history'        : hist_path,
        },
        'completed_at': datetime.datetime.now().isoformat(),
    }
    with open(os.path.join(cfg.results_dir,
              f'summary_member{member_id}.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)

    log.info(f"M{member_id} DONE  best={best_val:.5f}@ep{best_ep}  "
             f"wall={wall_total/60:.1f}min")
    return hist, summary


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    cfg = Config()
    os.makedirs(cfg.results_dir, exist_ok=True)
    log = setup_logging(cfg.results_dir)

    log.info("="*60)
    log.info("UNET V2 — IMPROVED SURROGATE TRAINING")
    log.info("="*60)
    log.info(f"Device      : {cfg.device}")
    if torch.cuda.is_available():
        log.info(f"GPU         : {torch.cuda.get_device_name(0)}")
        log.info(f"VRAM        : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    log.info(f"dataset_dir : {cfg.dataset_dir}")
    log.info(f"results_dir : {cfg.results_dir}")
    log.info(f"n_params    : {cfg.n_params}  (11 = 9 physical + t_rel + t_end_norm)")
    log.info(f"Loss weights: mse={cfg.w_mse} ssim={cfg.w_ssim} "
             f"grad={cfg.w_grad} tv={cfg.w_tv} front={cfg.w_front}")
    log.info(f"SSIM win    : {cfg.ssim_win_size}x{cfg.ssim_win_size}  "
             f"(was 11x11 in baseline — fixes 9px block artefact)")
    log.info(f"patience    : {cfg.patience}")
    log.info(f"Started     : {datetime.datetime.now().isoformat()}")

    # ── check dataset has 11-column X (t_rel + t_end_norm) ───────────────────
    x_path = os.path.join(cfg.dataset_dir, 'X_train.npy')
    x_test = np.load(x_path, mmap_mode='r')
    if x_test.shape[1] != 11:
        log.warning(
            f"X_train has {x_test.shape[1]} columns — expected 11.")
        log.warning(
            "The dataset was built with absolute time (10 cols).")
        log.warning(
            "Run rebuild_dataset_relative_time.py first to add t_rel + t_end_norm.")
        log.warning(
            "Falling back to n_params=10 for this run.")
        cfg.n_params = x_test.shape[1]

    # ── datasets ──────────────────────────────────────────────────────────────
    log.info("\nLoading datasets...")
    train_ds = FieldDataset(cfg.dataset_dir, 'train')
    val_ds   = FieldDataset(cfg.dataset_dir, 'val')
    log.info(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}")

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=True,
                          persistent_workers=cfg.num_workers > 0)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers, pin_memory=True,
                          persistent_workers=cfg.num_workers > 0)

    # ── train ensemble ────────────────────────────────────────────────────────
    summaries  = {}
    total_start = time.time()

    for m in range(cfg.n_members):
        ckpt_path    = os.path.join(cfg.results_dir, f'model_member{m}.pt')
        summary_path = os.path.join(cfg.results_dir, f'summary_member{m}.json')

        # skip if already trained (allows resume after job timeout)
        if os.path.exists(ckpt_path) and os.path.exists(summary_path):
            log.info(f"Member {m} already trained — skipping")
            with open(summary_path) as fh:
                s = json.load(fh)
            summaries[f'member{m}'] = s
            continue

        hist, summary = train_member(m, cfg.seeds[m], cfg, train_dl, val_dl)
        summaries[f'member{m}'] = summary

    total_wall = time.time() - total_start
    summaries['ensemble'] = {
        'total_wall_min': round(total_wall/60, 2),
        'best_member'   : min(range(cfg.n_members),
                              key=lambda m: summaries.get(
                                  f'member{m}', {}).get('best_val_loss', 1e9)),
        'completed_at'  : datetime.datetime.now().isoformat(),
    }
    with open(os.path.join(cfg.results_dir, 'ensemble_summary.json'), 'w') as fh:
        json.dump(summaries, fh, indent=2)

    log.info("="*60)
    log.info(f"ENSEMBLE COMPLETE  total={total_wall/60:.1f}min")
    for m in range(cfg.n_members):
        s = summaries.get(f'member{m}', {})
        log.info(f"  M{m}: best={s.get('best_val_loss','?'):.5f}"
                 f"@ep{s.get('best_epoch','?')}  "
                 f"wall={s.get('wall_time_min','?')}min")
    log.info(f"  best member: {summaries['ensemble']['best_member']}")
    log.info(f"\nFiles in {cfg.results_dir}:")
    for fn in sorted(os.listdir(cfg.results_dir)):
        sz = os.path.getsize(os.path.join(cfg.results_dir, fn))
        log.info(f"  {fn:50s}  {sz/1e6:.2f} MB")


if __name__ == '__main__':
    main()
