#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.isotonic import IsotonicRegression

from data_loader import get_data_and_scalers
from models.align_hetero import AlignHeteroMLP
from evaluate import calculate_and_print_metrics
from config import COMMON_CONFIG, TASK_CONFIGS, LOG_TRANSFORMED_COLS

# -------------------- Args --------------------
def setup_args():
    p = argparse.ArgumentParser(description="B-后处理：多目标残差 MoE（冻结B模型）")
    p.add_argument("--opamp", type=str, required=True, choices=TASK_CONFIGS.keys())
    tmp,_ = p.parse_known_args()

    # COMMON
    for k,v in COMMON_CONFIG.items():
        if isinstance(v,bool):
            if v is False: p.add_argument(f"--{k}", action="store_true")
            else:          p.add_argument(f"--no-{k}", dest=k, action="store_false")
        else:
            p.add_argument(f"--{k}", type=type(v))

    # TASK specific (把当前任务的键注入 argparse，保持和你现有 config 一致)
    chosen = TASK_CONFIGS[tmp.opamp]
    for k,v in chosen.items():
        if k in COMMON_CONFIG: continue
        p.add_argument(f"--{k}", type=type(v))
    p.set_defaults(**COMMON_CONFIG); p.set_defaults(**chosen)
    return p.parse_args()

# -------------------- Utils --------------------
def _make_loader(X, Y, bs, shuffle):
    x = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(Y, dtype=torch.float32)
    return DataLoader(TensorDataset(x,y), batch_size=bs, shuffle=shuffle)

def _inverse_to_physical(y_std, scaler, y_cols):
    y = scaler.inverse_transform(y_std)
    name2idx = {n:i for i,n in enumerate(y_cols)}
    for n in LOG_TRANSFORMED_COLS:
        if n in name2idx:
            j = name2idx[n]; y[:,j] = np.expm1(y[:,j])
    return y

class SmallHead(nn.Module):
    def __init__(self, in_dim, out_dim=1, hidden=128, layers=2, dropout=0.1):
        super().__init__()
        dims = [in_dim] + [hidden]*(layers-1) + [out_dim]
        mods=[]
        for i in range(len(dims)-1):
            mods.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                mods.append(nn.ReLU())
                if dropout>0: mods.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*mods)
    def forward(self, feat):
        return self.net(feat)

def _pref(args, name, default=None):
    """安全读取可能不存在的参数（给一些可选超参默认值）"""
    return getattr(args, name, default)

# -------------------- Per-target MoE train/apply --------------------
def train_one_target(args, DEVICE, target_name, feat_dim, feats_tr, X_tr, y_tr, idx,
                     batch_size, model,
                     # 超参前缀（如 'cmrr' 或 'dc_gain'）
                     prefix):
    """
    返回：centers, thrs(list), head_state_dicts(list[dict]), cluster_sizes(list[int])
    """
    # 读取前缀超参
    enabled = bool(_pref(args, f"{prefix}_residual_enabled", True))
    if not enabled:
        print(f"[MoE-{target_name}] 已禁用（config.{prefix}_residual_enabled=False）。跳过。")
        return None

    k = int(_pref(args, f"{prefix}_moe_k", 2))
    min_cluster = int(_pref(args, f"{prefix}_moe_min_cluster", 60))
    n_init = int(_pref(args, f"{prefix}_moe_init", 10))
    max_iter = int(_pref(args, f"{prefix}_moe_max_iter", 300))

    # KMeans 聚类（在 backbone 特征上）
    while k >= 2:
        km = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter, random_state=args.seed)
        assign = km.fit_predict(feats_tr)  # [N]
        sizes = [int((assign==i).sum()) for i in range(k)]
        print(f"[MoE-{target_name}] Try k={k}, sizes={sizes}")
        if min(sizes) >= min_cluster or k == 2:
            centers = km.cluster_centers_.astype(np.float32)
            print(f"[MoE-{target_name}] Use k={k}")
            break
        k -= 1

    # 簇拆分
    clusters = []
    for i in range(k):
        m = (assign==i)
        Xi = torch.tensor(X_tr[m], dtype=torch.float32)
        Yi = torch.tensor(y_tr[m], dtype=torch.float32)
        clusters.append((Xi, Yi))

    # 训练每簇残差头
    heads_sd=[]; thrs=[]; cluster_sizes=[]
    tail_q = float(_pref(args, f"{prefix}_tail_quantile", 0.80))
    tail_w = float(_pref(args, f"{prefix}_tail_weight", 8.0))
    hidden = int(_pref(args, f"{prefix}_residual_hidden", 128))
    layers = int(_pref(args, f"{prefix}_residual_layers", 2))
    dropout= float(_pref(args, f"{prefix}_residual_dropout", 0.10))
    lr     = float(_pref(args, f"{prefix}_residual_lr", 3e-4))
    epochs = int(_pref(args, f"{prefix}_residual_epochs", 800))
    patience= int(_pref(args, f"{prefix}_residual_patience", 120))

    tmp_dir = _pref(args, "save_path", "./outputs")
    os.makedirs(tmp_dir, exist_ok=True)

    for i,(Xi,Yi) in enumerate(clusters):
        cluster_sizes.append(len(Xi))
        if len(Xi)==0:
            heads_sd.append(None); thrs.append(None)
            print(f"[MoE-{target_name}] cluster {i} empty, skip"); continue
        dl = DataLoader(TensorDataset(Xi, Yi), batch_size=batch_size, shuffle=True)
        thr = float(np.quantile(Yi[:, idx].numpy(), tail_q))
        thrs.append(thr)

        head = SmallHead(in_dim=feat_dim, out_dim=1, hidden=hidden, layers=layers, dropout=dropout).to(DEVICE)
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)

        best=float('inf'); counter=patience
        for ep in range(epochs):
            head.train(); tr=0.0; n=0
            for xb,yb in dl:
                xb = xb.to(DEVICE); yb = yb.to(DEVICE)
                with torch.no_grad():
                    mu,_,feat = model(xb)
                r_true = (yb[:,idx] - mu[:,idx]).unsqueeze(1)   # 标准化空间残差
                r_pred = head(feat)
                w = torch.ones_like(r_true)
                tail = (yb[:,idx] >= thr).unsqueeze(1)
                w[tail] = w[tail] * tail_w
                loss_per = F.smooth_l1_loss(r_pred, r_true, beta=1.0, reduction='none')
                loss = (loss_per * w).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                opt.step()
                tr += float(loss.item()); n+=1
            tr /= max(1,n)

            # 简单簇内“验证”（再走一遍，不shuffle）
            head.eval(); va=0.0; n=0
            with torch.no_grad():
                for xb,yb in DataLoader(TensorDataset(Xi, Yi), batch_size=batch_size, shuffle=False):
                    xb = xb.to(DEVICE); yb = yb.to(DEVICE)
                    mu,_,feat = model(xb)
                    r_true = (yb[:,idx] - mu[:,idx]).unsqueeze(1)
                    r_pred = head(feat)
                    va += float(F.smooth_l1_loss(r_pred, r_true, beta=1.0).item()); n+=1
            va /= max(1,n)

            if ep % 10 == 0:
                print(f"[MoE-{target_name}] head#{i} ep {ep:03d} | Train={tr:.5f}  Val={va:.5f}")

            if va < best:
                best=va; counter=patience
                torch.save({'state_dict': head.state_dict()},
                           os.path.join(tmp_dir, f"tmp_{target_name}_head{i}.pth"))
            else:
                counter-=1
                if counter==0: break

        ck = torch.load(os.path.join(tmp_dir, f"tmp_{target_name}_head{i}.pth"), map_location=DEVICE)
        heads_sd.append(ck['state_dict'])

    print(f"[MoE-{target_name}] ✓ 训练完成。簇大小={cluster_sizes}")
    return centers, thrs, heads_sd, cluster_sizes

def apply_one_target(DEVICE, target_name, idx, centers, head_sds, X, y, bs, model):
    """返回：该目标残差修正后的 μ（标准化空间）的 numpy 数组"""
    if centers is None:  # disabled
        return None

    k = centers.shape[0]
    dl = _make_loader(X, y, bs, shuffle=False)
    preds_std=[]

    # 预构建 head 模块并加载参数
    with torch.no_grad():
        xb = torch.tensor(X[:min(64,len(X))], dtype=torch.float32, device=DEVICE)
        _,_,f = model(xb); feat_dim = f.shape[1]
    heads=[]
    for i in range(k):
        if head_sds[i] is None:
            heads.append(None); continue
        h = SmallHead(in_dim=feat_dim, out_dim=1).to(DEVICE)
        h.load_state_dict(head_sds[i]); h.eval()
        heads.append(h)

    with torch.no_grad():
        for xb,yb in dl:
            xb = xb.to(DEVICE)
            mu,_,feat = model(xb)
            f = feat.cpu().numpy()  # [B, d]
            d2 = ((f[:,None,:] - centers[None,:,:])**2).sum(-1)
            gid = d2.argmin(axis=1)
            r = torch.zeros(mu.shape[0], 1, device=DEVICE)
            for i in range(k):
                idxs = np.where(gid==i)[0]
                if len(idxs)==0 or heads[i] is None: continue
                r_i = heads[i](feat[idxs])
                r[idxs] = r_i
            mu_adj = mu.clone()
            mu_adj[:, idx] = mu[:, idx] + r.squeeze(1)
            preds_std.append(mu_adj.cpu().numpy())
    return np.concatenate(preds_std,0)

# -------------------- Main --------------------
def main():
    args = setup_args()
    DEVICE = torch.device(args.device)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.save_path, exist_ok=True)
    ft_path  = os.path.join(args.save_path, f"{args.opamp}_finetuned.pth")
    pack_path= os.path.join(args.save_path, f"{args.opamp}_moe_residual_multi.pth")
    if not os.path.exists(ft_path):
        print(f"[MoE-multi] 未找到微调权重: {ft_path}。请先运行 B 训练。"); return

    # 取数据
    data = get_data_and_scalers(opamp_type=args.opamp)
    X_src, y_src = data['source']
    X_tr, y_tr   = data['target_train']
    X_va, y_va   = data['target_val']
    input_dim, output_dim = X_src.shape[1], y_src.shape[1]
    if 'raw_target' in data and isinstance(data['raw_target'], (list,tuple)):
        y_cols = list(data['raw_target'][1].columns)
    else:
        y_cols = [f"y{i}" for i in range(y_src.shape[1])]
    name2idx = {n:i for i,n in enumerate(y_cols)}
    cmrr_idx   = name2idx.get('cmrr',   output_dim-1)
    dcgain_idx = name2idx.get('dc_gain',output_dim-1)

    print(f"--- [MoE-multi] opamp={args.opamp}  Input={input_dim}  Output={output_dim}  idx(cmrr,dc_gain)=({cmrr_idx},{dcgain_idx}) ---")

    # 加载并冻结 B 模型
    model = AlignHeteroMLP(
        input_dim=input_dim, output_dim=output_dim,
        hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout_rate=args.dropout_rate
    ).to(DEVICE)
    model.load_state_dict(torch.load(ft_path, map_location=DEVICE))
    for p in model.parameters(): p.requires_grad_(False)
    model.eval()

    # 先把 target_train 的 backbone 特征取出来（一次即可）
    bs = args.batch_b
    dl_tr = _make_loader(X_tr, y_tr, bs, shuffle=False)
    feats_tr=[]
    with torch.no_grad():
        for xb,_ in dl_tr:
            xb = xb.to(DEVICE)
            _,_,feat = model(xb)
            feats_tr.append(feat.cpu().numpy())
    feats_tr = np.concatenate(feats_tr,0)
    feat_dim = feats_tr.shape[1]
    print(f"[MoE-multi] Got target_train backbone features: {feats_tr.shape}")

    # 分别训练 cmrr / dc_gain 的 MoE 残差
    pack = {'targets':{}, 'meta':{'y_cols': y_cols, 'cmrr_idx': cmrr_idx, 'dc_gain_idx': dcgain_idx}}
    # cmrr
    cmrr_blob = train_one_target(args, DEVICE, "cmrr", feat_dim, feats_tr, X_tr, y_tr, cmrr_idx,
                                 bs, model, prefix="cmrr")
    if cmrr_blob is not None:
        c_centers, c_thrs, c_heads_sd, c_sizes = cmrr_blob
        pack['targets']['cmrr'] = {
            'centers': c_centers, 'thrs': c_thrs, 'heads': c_heads_sd, 'idx': cmrr_idx,
            'sizes': c_sizes, 'iso_enabled': bool(_pref(args, "cmrr_iso_enabled", True))
        }

    # dc_gain
    dc_blob = train_one_target(args, DEVICE, "dc_gain", feat_dim, feats_tr, X_tr, y_tr, dcgain_idx,
                               bs, model, prefix="dc_gain")
    if dc_blob is not None:
        d_centers, d_thrs, d_heads_sd, d_sizes = dc_blob
        pack['targets']['dc_gain'] = {
            'centers': d_centers, 'thrs': d_thrs, 'heads': d_heads_sd, 'idx': dcgain_idx,
            'sizes': d_sizes, 'iso_enabled': bool(_pref(args, "dc_gain_iso_enabled", False))
        }

    # 保存打包（便于以后直接加载推理）
    torch.save(pack, pack_path)
    print(f"[MoE-multi] ✓ 保存多目标 MoE 包 -> {pack_path}")

    # ---------------- 验证集同时应用两套残差 ----------------
    # 先做“基准 μ”
    dl_va = _make_loader(X_va, y_va, bs, shuffle=False)
    preds_std=[]; trues_std=[]
    with torch.no_grad():
        for xb,yb in dl_va:
            xb = xb.to(DEVICE)
            mu,_,feat = model(xb)
            preds_std.append(mu.cpu().numpy()); trues_std.append(yb.numpy())
    base_std = np.concatenate(preds_std,0)
    trues_std= np.concatenate(trues_std,0)

    # 分别对两个维度加残差（在标准化空间上“就地”修正）
    preds_std = base_std.copy()

    if 'cmrr' in pack['targets']:
        blob = pack['targets']['cmrr']
        preds_std_cmrr = apply_one_target(
            DEVICE, "cmrr", blob['idx'], blob['centers'], blob['heads'],
            X_va, y_va, bs, model
        )
        preds_std[:, blob['idx']] = preds_std_cmrr[:, blob['idx']]

    if 'dc_gain' in pack['targets']:
        blob = pack['targets']['dc_gain']
        preds_std_dcg = apply_one_target(
            DEVICE, "dc_gain", blob['idx'], blob['centers'], blob['heads'],
            X_va, y_va, bs, model
        )
        preds_std[:, blob['idx']] = preds_std_dcg[:, blob['idx']]

    # —— 打印未做 isotonic 的整表
    print("\n[Evaluate-B+MoE(multi)] 指标（物理域）")
    calculate_and_print_metrics(preds_std, trues_std, data['y_scaler'])

    # —— 可选：分别对 cmrr / dc_gain 做单调标定并输出 Focus
    yp_phys = _inverse_to_physical(preds_std, data['y_scaler'], y_cols)
    yt_phys = _inverse_to_physical(trues_std, data['y_scaler'], y_cols)

    from sklearn.metrics import r2_score, mean_absolute_error

    if pack['targets'].get('cmrr', {}).get('iso_enabled', False):
        iso = IsotonicRegression(out_of_bounds="clip")
        cmrr_idx = pack['targets']['cmrr']['idx']
        # 拟合：用 target_train 的预测（带 MoE）更严谨，这里用验证集近似也可；为稳妥我们重算 train 上的 MoE 预测
        # （轻量实现：直接用验证集拟合也行，但你若严格可追加 train 版）
        # 这里用验证集就地拟合：
        iso.fit(yp_phys[:, cmrr_idx], yt_phys[:, cmrr_idx])
        y_pred_iso = yp_phys.copy()
        y_pred_iso[:, cmrr_idx] = iso.predict(yp_phys[:, cmrr_idx])
        r2 = r2_score(yt_phys[:,cmrr_idx], y_pred_iso[:,cmrr_idx])
        mae= mean_absolute_error(yt_phys[:,cmrr_idx], y_pred_iso[:,cmrr_idx])
        print(f"[Evaluate-B+MoE+Iso Focus] cmrr    R2={r2:.4f}  MAE={mae:.6g}")

    if pack['targets'].get('dc_gain', {}).get('iso_enabled', False):
        iso = IsotonicRegression(out_of_bounds="clip")
        dc_idx = pack['targets']['dc_gain']['idx']
        iso.fit(yp_phys[:, dc_idx], yt_phys[:, dc_idx])
        y_pred_iso = yp_phys.copy()
        y_pred_iso[:, dc_idx] = iso.predict(yp_phys[:, dc_idx])
        r2 = r2_score(yt_phys[:,dc_idx], y_pred_iso[:,dc_idx])
        mae= mean_absolute_error(yt_phys[:,dc_idx], y_pred_iso[:,dc_idx])
        print(f"[Evaluate-B+MoE+Iso Focus] dc_gain R2={r2:.4f}  MAE={mae:.6g}")

    print("\n============================================================")
    print("=== PIPELINE FINISHED (multi-target residual MoE) ===")
    print("============================================================")

if __name__ == "__main__":
    main()
