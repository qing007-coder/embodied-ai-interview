import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleHeadCrossAttention(nn.Module):
    """
    单头交叉注意力
    Q 来自Decoder隐状态
    K, V 来自Encoder的memory
    """
    def __init__(self, q_dim, kv_dim, d_dim, bias=True, dropout=0.1):
        super().__init__()
        # q_in_dim: decoder query输入维度
        # kv_in_dim: encoder memory(K/V)输入维度
        # d_dim: 注意力头维度，缩放因子 sqrt(d_dim)
        self.d_k = d_dim

        self.q_proj = nn.Linear(q_dim, d_dim, bias=bias)
        self.k_proj = nn.Linear(kv_dim, d_dim, bias=bias)
        self.v_proj = nn.Linear(kv_dim, d_dim, bias=bias)
        self.o_proj = nn.Linear(d_dim, d_dim, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B, Lq, _ = q.shape
        Bk, Lk, _ = k.shape
        assert B == Bk, "Batch size 不一致！"

        # 投影
        q = self.q_proj(q)  # [B, Lq, d_dim]
        k = self.k_proj(k)  # [B, Lk, d_dim]
        v = self.v_proj(v)  # [B, Lk, d_dim]

        # 注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_dim ** 0.5)  # [B, Lq, Lk]

        if mask is not None:
            mask = mask.to(scores.device)
            if mask.dtype == torch.bool:
                # mask=True: 可见；mask=False: 遮蔽，填充极小值
                fill_val = torch.finfo(scores.dtype).min
                scores = scores.masked_fill(~mask, fill_val)
            else:
                # 加法式mask（比如 -1e9 形式）
                scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1)  # [B, Lq, Lk]
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)  # [B, Lq, d_dim]
        out = self.o_proj(out)
        out = self.out_dropout(out)

        return out, attn_weights
