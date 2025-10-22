# src/losses.py
import torch
import torch.nn.functional as F

def heteroscedastic_nll(mu, logvar, y, reduction='mean'):
    # logvar = log(σ^2)
    inv_var = torch.exp(-logvar)
    nll = 0.5 * (inv_var * (y - mu)**2 + logvar)
    if reduction == 'mean':
        return nll.mean()
    elif reduction == 'sum':
        return nll.sum()
    else:
        return nll

def batch_r2(y_true, y_pred, eps=1e-8):
    # 按列计算 batch R^2
    y_true_mean = y_true.mean(dim=0, keepdim=True)
    ss_tot = ((y_true - y_true_mean)**2).sum(dim=0)
    ss_res = ((y_true - y_pred)**2).sum(dim=0)
    r2 = 1.0 - ss_res / (ss_tot + eps)
    return r2  # [D]

def coral_loss(feat_a, feat_b, unbiased=True, eps=1e-6):
    """
    轻量CORAL：均值与协方差对齐。
    输入: feat_*: [B, D]
    """
    def _cov(x):
        x = x - x.mean(dim=0, keepdim=True)
        denom = (x.size(0) - 1) if unbiased else x.size(0)
        c = (x.T @ x) / (denom + eps)
        return c
    ma, mb = feat_a.mean(0), feat_b.mean(0)
    Ca, Cb = _cov(feat_a), _cov(feat_b)
    mean_term = (ma - mb).pow(2).mean()
    cov_term  = (Ca - Cb).pow(2).mean()
    return mean_term + cov_term
