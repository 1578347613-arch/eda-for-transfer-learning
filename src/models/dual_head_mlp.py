# models/dual_head_mlp.py

import torch
import torch.nn as nn

# --- 移除了所有对 config 的依赖 ---

__all__ = [
    "DualHeadMLP",
    "copy_from_single_head_to_dualhead",
    "l2sp_regularizer",
]

class DualHeadMLP(nn.Module):
    """
    双头 MLP：
      - backbone: 到隐藏层为止（不含最终的 hidden->output 线性层）
      - head_A: hidden_dim -> output_dim（源域 / 基线头）
      - head_B: hidden_dim -> output_dim（目标域 / 微调头）

    forward(x, domain):
      - domain='A' 或 'B'，分别走 head_A / head_B
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,    # 变为必需参数
        num_layers: int,    # 变为必需参数
        dropout_rate: float # 变为必需参数
    ):
        super().__init__()
        
        # --- 移除了 if is None 和 getattr(config, ...) 的逻辑 ---

        layers = []
        # 输入层: input_dim -> hidden_dim
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout_rate and dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))

        # 隐藏层: (num_layers - 1) 次 hidden -> hidden
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate and dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

        # 注意：不添加 hidden->output 的线性层，交由双头
        self.backbone = nn.Sequential(*layers)

        # 两个输出头
        self.head_A = nn.Linear(hidden_dim, output_dim)
        self.head_B = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, domain: str = "A") -> torch.Tensor:
        feats = self.backbone(x)
        d = (domain or "A").lower()
        if d in ("a", "src", "source"):
            return self.head_A(feats)
        elif d in ("b", "tgt", "target"):
            return self.head_B(feats)
        else:
            raise ValueError(f"Unknown domain '{domain}'. Use 'A' or 'B'.")


@torch.no_grad()
def copy_from_single_head_to_dualhead(single_mlp: nn.Module, dual: DualHeadMLP):
    """
    把单头 MLP（models.mlp.MLP）的权重复制到双头：
      1) single_mlp.network 的所有隐藏层（不含最后 output 层） -> dual.backbone
      2) single_mlp 最后一层 Linear（hidden->output） -> dual.head_A & dual.head_B
    """
    # single 的线性层（包含最后的 output 层）
    single_linear = [m for m in single_mlp.network if isinstance(m, nn.Linear)]
    # dual 的线性层（只有 backbone 的）
    dual_linear   = [m for m in dual.backbone    if isinstance(m, nn.Linear)]

    if len(single_linear) < len(dual_linear) + 1:
        raise RuntimeError(
            f"[copy] 单头 Linear 数量不足：single={len(single_linear)}，"
            f"dual_backbone 需要 {len(dual_linear)} + 输出层 1"
        )

    # 1) 复制 backbone 的线性层（不含 single 的最后一层）
    for i in range(len(dual_linear)):
        if (dual_linear[i].weight.shape != single_linear[i].weight.shape or
            dual_linear[i].bias.shape   != single_linear[i].bias.shape):
            raise RuntimeError(
                f"[copy] 第 {i} 个线性层形状不匹配："
                f"single={tuple(single_linear[i].weight.shape)} vs "
                f"dual={tuple(dual_linear[i].weight.shape)}"
            )
        dual_linear[i].weight.copy_(single_linear[i].weight)
        dual_linear[i].bias.copy_(single_linear[i].bias)

    # 2) 用 single 的最后一个输出层初始化双头
    last = single_linear[-1]
    if dual.head_A.weight.shape != last.weight.shape or dual.head_A.bias.shape != last.bias.shape:
        raise RuntimeError(
            f"[copy] 输出头形状不匹配：single_last={tuple(last.weight.shape)}, "
            f"head_A={tuple(dual.head_A.weight.shape)}"
        )

    dual.head_A.weight.copy_(last.weight)
    dual.head_A.bias.copy_(last.bias)
    dual.head_B.weight.copy_(last.weight)
    dual.head_B.bias.copy_(last.bias)

    return dual


def l2sp_regularizer(model: nn.Module, pretrained_state: dict, scale: float = 1e-4) -> torch.Tensor:
    """
    L2-SP 正则：对 backbone 参数施加 ||θ - θ*||^2 约束
      - pretrained_state：通常是 copy.deepcopy(model.state_dict())，表示 θ*
      - 只对 name 以 'backbone.' 开头且 requires_grad=True 的参数做正则
    """
    device = next(model.parameters()).device
    reg = torch.zeros((), device=device)

    for name, p in model.named_parameters():
        if (not p.requires_grad) or (not name.startswith("backbone.")):
            continue
        ref = pretrained_state.get(name, None)
        if isinstance(ref, torch.Tensor) and ref.shape == p.shape:
            reg = reg + (p - ref.to(device)).pow(2).sum()

    return reg * scale

