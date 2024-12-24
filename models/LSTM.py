import torch.nn as nn
from models.model_v6 import Dense

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_sizes, pred_len, dropout, **kwargs):
        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_sizes[0], batch_first=True)
        self.dropout = nn.Dropout(dropout)  # 设置 Dropout 层
        self.lstm2 = nn.LSTM(input_size=hidden_sizes[0], hidden_size=hidden_sizes[1], batch_first=True)
        self.fc_soh = Dense(hidden_sizes[1], pred_len)
        self.fc_rul = Dense(hidden_sizes[1], 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        out = out[:, -1, :]
        return self.fc_soh(out), self.fc_rul(out).squeeze(-1)