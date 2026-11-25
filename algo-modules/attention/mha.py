import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Pre-LN 结构里  LayerNorm 属于 block  不属于 attention 模块"""

    def __init__(self, num_heads, q_dim, k_dim=None, v_dim=None, dropout=0.0, bias=True):
        super().__init__()
        k_dim = q_dim if k_dim is None else k_dim
        v_dim = q_dim if v_dim is None else v_dim
        assert q_dim % num_heads == 0, "q_dim 必须能被 num_heads 整除"

        self.num_heads = num_heads
        self.head_dim = q_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # q/k/v 一律投到 num_heads*head_dim = q_dim 维，o_proj 投回 q_dim
        self.q_proj = nn.Linear(q_dim, q_dim, bias=bias)
        self.k_proj = nn.Linear(k_dim, q_dim, bias=bias)
        self.v_proj = nn.Linear(v_dim, q_dim, bias=bias)
        self.o_proj = nn.Linear(q_dim, q_dim, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None, is_causal=False):
        """
        q: (B, Lq, q_dim);  k/v: (B, Lk, k_dim/v_dim)
        mask: bool 张量 True 表示该位置允许 attend
              形状 (B,1,Lq,Lk)/(1,1,Lq,Lk)/(Lq,Lk)，可广播；
              也接受 float 加法掩码。
        """
        B, Lq, _ = q.shape
        Lk = k.shape[1]
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(q).view(B, Lq, H, D).transpose(1, 2)   # (B,H,Lq,D)
        k = self.k_proj(k).view(B, Lk, H, D).transpose(1, 2)   # (B,H,Lk,D)
        v = self.v_proj(v).view(B, Lk, H, D).transpose(1, 2)   # (B,H,Lk,D)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if is_causal:
            causal = torch.tril(torch.ones(Lq, Lk, device=q.device)) > 0
            mask = causal if mask is None else (mask & causal)

        if mask is not None:
            if mask.dtype == torch.bool:
                min_val = torch.finfo(scores.dtype).min
                scores = scores.masked_fill(~mask, min_val)
            else:                                  # 已是加法掩码
                scores = scores + mask

        weights = self.attn_dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(weights, v)                          # (B,H,Lq,D)
        out = out.transpose(1, 2).reshape(B, Lq, H * D)         # (B,Lq,q_dim)
        return self.out_dropout(self.o_proj(out))
