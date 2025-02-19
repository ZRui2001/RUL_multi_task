"""
结合时序位置编码（model_v2_1）、因果注意力（model_v3）
"""

import torch
import torch.nn as nn
import math

class model_v5(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, pred_len, dropout=0.1, **kwargs):
        """
        带因果注意力的 Transformer 模型。
        """
        super(model_v5, self).__init__()
        
        # 输入嵌入层
        self.input_embedding = nn.Linear(input_dim, model_dim - 1)
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=num_heads, 
            dim_feedforward=model_dim * 4, 
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.fc_soh = nn.Linear(model_dim, pred_len)
        self.fc_rul = nn.Linear(model_dim, 1)

    def forward(self, x):
        """
        参数:
        - x: 输入张量，形状为 (batch_size, lookback_len, 1)
        
        返回:
        - soh_pred，形状为 (batch_size, pred_len)
        - rul_pred，形状为 (batch_size,)
        """
        batch_size, lookback_len, _ = x.shape
        device = x.device

        # 输入嵌入与位置编码
        x = self.input_embedding(x)  # (batch_size, lookback_len, model_dim)
        
        # 动态位置编码
        position = torch.arange(lookback_len).unsqueeze(0).repeat(batch_size, 1).float() / lookback_len
        position = position.unsqueeze(2).to(x.device)  # (batch_size, lookback_len, 1)
        x = torch.cat([x, position], dim=2)  # (batch_size, lookback_len, model_dim)

        # 因果注意力掩码
        causal_mask = generate_causal_mask(seq_len=lookback_len, device=device)  # (lookback_len, lookback_len)

        # Transformer 编码器
        x = self.encoder(x, mask=causal_mask)  # mask 参数应用因果掩码
        
        # 取最后一个时间步的输出
        x = x[:, -1, :].squeeze(1)  # (batch_size, model_dim)
        
        # 输出预测
        return self.fc_soh(x), self.fc_rul(x).squeeze(-1)  # (batch_size, pred_len), (batch_size,)

    
def generate_causal_mask(seq_len, device):
    """
    生成因果注意力掩码
    :param seq_len: 序列长度
    :param device: 当前设备
    :return: 因果掩码 (seq_len, seq_len)
    """
    # 上三角掩码 (j > i) 的部分设置为 -inf
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
    return mask
