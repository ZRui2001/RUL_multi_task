import torch.nn as nn
from models.model_v6 import Dense

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, pred_len, num_layers, dropout, **kwargs):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc_soh = Dense(hidden_size, pred_len)
        self.fc_rul = Dense(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc_soh(out), self.fc_rul(out).squeeze(-1)