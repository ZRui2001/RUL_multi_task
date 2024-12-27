import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class model_v6(nn.Module):
    def __init__(self, input_dim, model_dim, lookback_len, num_heads, num_heads_s, num_layers, depth, kernel_size, pred_len, dropout=0.1, **kwargs):
        super(model_v6, self).__init__()
        self.emb_t = nn.Linear(input_dim, model_dim - 1)
        self.emb_s = nn.Linear(input_dim, model_dim)
        self.drop_t = nn.Dropout(dropout)
        self.drop_s = nn.Dropout(dropout)
        self.dtpe = DynamicTemporalPositionalEncoding(dropout=dropout)
        self.pe = PositionalEncoding(model_dim=lookback_len, dropout=dropout)
        self.layers = nn.ModuleList([
            SpatiotemporalAttentionBlock(
                model_dim=model_dim,
                lookback_len=lookback_len,
                num_heads=num_heads,
                num_heads_s=num_heads_s,
                kernel_size=kernel_size,
                depth=depth,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        self.gated_fusion = GatedFusion(model_dim=model_dim)
        self.dense_soh = Dense(model_dim=model_dim, out_dim=pred_len)
        self.dense_rul = Dense(model_dim=model_dim, out_dim=1)

    def forward(self, x):
        """
        x.shape = (b, t, 1)
        """
        x_t = self.drop_t(self.emb_t(x))  # (b, t, d-1)
        x_s = self.drop_s(self.emb_s(x)) # (b, t, d)

        x_t = self.dtpe(x_t)  # (b, t, d)
        # (B,T,D) --transpose-> (B,D,T)
        x_s = self.pe(x_s.transpose(1, 2))

        for layer in self.layers:
            x_t, x_s = layer(x_t, x_s)

        x = self.gated_fusion(x_t, x_s.transpose(1, 2))

        # 取最后时间步输出
        x = x[:, -1, :].squeeze(1)   # (b, d)
        soh_pred = self.dense_soh(x)  # (b, pred_len)
        rul_pred = self.dense_rul(x).squeeze(-1)   # (b,)
        return soh_pred, rul_pred


class DynamicTemporalPositionalEncoding(nn.Module):
    def __init__(self, dropout):
        super(DynamicTemporalPositionalEncoding, self).__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x.shape = (b, t, d-1)
        """
        batch_size, lookback_len, _ = x.shape
        position = torch.arange(lookback_len).unsqueeze(0).repeat(batch_size, 1).float() / lookback_len
        position = position.unsqueeze(2).to(x.device)  # (b, t, 1)

        position = position.transpose(1, 2)  # (b, 1, t)
        position = self.conv(position)  # (b, 1, t)
        position = self.softmax(position).transpose(1, 2)   # (b, t, 1)

        x = torch.cat([x, position], dim=2)  # (b, t, d)
        return self.dropout(x)  # (b, t, d)


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2) * (-math.log(10000.0) / model_dim))
        pe = torch.zeros(max_len, model_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, model_dim)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        In: x.shape = (B,D,T)
        Out: x.shape = (B,D,T)
        """
        _, D, _ = x.shape
        x = x + self.pe[:, :D, :]
        return self.dropout(x)


class SpatiotemporalAttentionBlock(nn.Module):
    def __init__(self, model_dim, lookback_len, num_heads, num_heads_s, kernel_size, depth, dropout=0.1):
        super(SpatiotemporalAttentionBlock, self).__init__()
        self.dcmsa = DCMSABlock(model_dim=model_dim, num_heads=num_heads, kernel_size=kernel_size, depth=depth, dropout=dropout)
        self.imsa = IMSA(lookback_len=lookback_len, num_heads=num_heads_s, dropout=dropout)
        self.caf = CrossAttentionFusion(model_dim=model_dim, num_heads=num_heads, dropout=dropout)

    def forward(self, x_t, x_s):
        x_t = self.dcmsa(x_t)
        x_s = self.imsa(x_s)
        x_t, x_s = self.caf(x_t, x_s)
        return x_t, x_s


class DCMSABlock(nn.Module):
    def __init__(self, model_dim, num_heads, kernel_size, depth, dropout=0.1):
        super(DCMSABlock, self).__init__()
        self.layers = nn.ModuleList([])
        for d in range(depth):
            self.layers.append(nn.ModuleList([
                DCMSA(model_dim=model_dim, num_heads=num_heads, kernel_size=kernel_size, dilation=2**d, dropout=dropout),
                FFN(model_dim=model_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ffn in self.layers:
            x = attn(x) + x
            x = ffn(x) + x
        return x


class DCMSA(nn.Module):
    def __init__(self, model_dim, num_heads, kernel_size, dilation, dropout=0.1):
        super(DCMSA, self).__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.scale = (model_dim // num_heads) ** -0.5

        self.norm = nn.LayerNorm(model_dim)
        self.qkv = nn.Linear(model_dim, model_dim * 3, bias=False)
        self.proj = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape

        x = self.norm(x)

        # (B, T, D) --qkv-> (B, T, 3D) --reshape-> (B, T, 3, h, d) --permute-> (3, B, h, T, d)  D=h*d
        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, D //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        # (B, h, T, d)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # (B,h,T,d) @ (B,h,d,T) = (B,h,T,T)
        attn = q @ k.transpose(-2, -1) * self.scale

        # 对于每个Query，以dilation间隔向前保留kernel_size个有效Key（包括本身）
        mask = torch.full((T, T), float("-inf"), device=x.device)
        for i in range(T):
            for j in range(self.kernel_size):
                idx = i - j * self.dilation
                if idx >= 0:
                    mask[i, idx] = 0
        attn += mask.unsqueeze(0).unsqueeze(0)

        x = (F.softmax(attn, dim=-1) @ v).transpose(1, 2).reshape(B, T, D)

        x = self.proj(x)
        x = self.dropout(x)
        return x


class FFN(nn.Module):
    def __init__(self, model_dim, dropout=0.1):
        super(FFN, self).__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class IMSA(nn.Module):
    def __init__(self, lookback_len, num_heads, dropout=0.1):
        super(IMSA, self).__init__()
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(lookback_len)
        self.qkv = nn.Linear(lookback_len, lookback_len*3, bias=False)
        self.scale = (lookback_len // num_heads) ** -0.5
        self.proj = nn.Linear(lookback_len, lookback_len)
        self.dropout = nn.Dropout(dropout)
        self.ffn = FFN(model_dim=lookback_len, dropout=dropout)

    def forward(self, x):
        """
        In:
            - x: (B,D,T)
        Out:
            - x: (B,D,T)
        """
        B, D, T = x.shape

        x_norm = self.norm(x)

        # (B,D,T) --qkv-> (B,D,3T) --reshape-> (B,D,3,h,t) --permute-> (3,B,h,D,t)  T=h*t
        qkv = self.qkv(x_norm).reshape(B, D, 3, self.num_heads, T // self.num_heads).permute(2, 0, 3, 1, 4)
        # (B,h,D,t)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # (B,h,D,t) @ (B,h,t,D) = (B,h,D,D)
        attn = q @ k.transpose(-2, -1) * self.scale
        # (B,h,D,D) @ (B,h,D,t)
        out = (F.softmax(attn, dim=-1) @ v).transpose(1, 2).reshape(B, D, T)

        out = self.proj(out)
        x = self.dropout(out) + x
        x = self.ffn(x) + x
        return x


class CrossAttentionFusion(nn.Module):
    def __init__(self, model_dim, num_heads, dropout=0.1):
        super(CrossAttentionFusion, self).__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.scale = (model_dim // num_heads) ** -0.5

        self.norm_t = nn.LayerNorm(model_dim)
        self.norm_s = nn.LayerNorm(model_dim)
        self.qkv_t = nn.Linear(model_dim, model_dim * 3, bias=False)
        self.qkv_s = nn.Linear(model_dim, model_dim * 3, bias=False)
        self.proj_t = nn.Linear(model_dim, model_dim)
        self.proj_s = nn.Linear(model_dim, model_dim)
        self.drop_t = nn.Dropout(dropout)
        self.drop_s = nn.Dropout(dropout)
        self.ffn_t = FFN(model_dim=model_dim, dropout=dropout)
        self.ffn_s = FFN(model_dim=model_dim, dropout=dropout)

    def forward(self, x_t, x_s):
        """
        In:
            - x_t: (B,T,D)
            - x_s: (B,D,T)
        Out:
            - x_t: (B,T,D)
            - x_s: (B,D,T)
        """
        B, T, D = x_t.shape

        x_t = self.norm_t(x_t)
        # (B,D,T) --permute-> (B,T,D)
        x_s = self.norm_s(x_s.transpose(1, 2))

        # (B,T,D) --qkv-> (B,T,3D) --reshape-> (B,T,3,h,d) --permute-> (3,B,h,T,d)  D=h*d
        qkv_t = self.qkv_t(x_t).reshape(B, T, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv_s = self.qkv_s(x_s).reshape(B, T, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        # (B,h,T,d)
        q_t, k_t, v_t = qkv_t[0], qkv_t[1], qkv_t[2]
        q_s, k_s, v_s = qkv_s[0], qkv_s[1], qkv_s[2]

        # 信息交互，(B,h,T,d) @ (B,h,d,T) = (B,h,T,T)
        attn_t = q_s @ k_t.transpose(-1, -2) * self.scale
        attn_s = q_t @ k_s.transpose(-1, -2) * self.scale

        x_t = (F.softmax(attn_t, dim=-1) @ v_t).transpose(1, 2).reshape(B, T, D)
        x_s = (F.softmax(attn_s, dim=-1) @ v_s).transpose(1, 2).reshape(B, T, D)

        x_t = self.ffn_t(self.drop_t(self.proj_t(x_t)))
        x_s = self.ffn_s(self.drop_s(self.proj_s(x_s))).transpose(1, 2)
        return x_t, x_s


class GatedFusion(nn.Module):
    def __init__(self, model_dim):
        super(GatedFusion, self).__init__()
        self.norm_t = nn.LayerNorm(model_dim)
        self.norm_s = nn.LayerNorm(model_dim)
        self.fc_t = nn.Linear(model_dim, model_dim)
        self.fc_s = nn.Linear(model_dim, model_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_t, x_s):
        """
        In:
            - x_t: (B,T,D)
            - x_s: (B,T,D)
        Out:
            - x: (B,T,D)
        """
        x_t = self.norm_t(x_t)
        x_s = self.norm_s(x_s)

        z = self.sigmoid(self.fc_t(x_t) + self.fc_s(x_s))

        x = z * x_t + (1 - z) * x_s
        return x


class Dense(nn.Module):
    def __init__(self, model_dim, out_dim):
        super(Dense, self).__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Linear(model_dim * 4, out_dim)
        )

    def forward(self, x):
        """
        In:
            - x: (B,D)
        Out:
            - out: (B,L)
        """
        x = self.net(x)
        return x