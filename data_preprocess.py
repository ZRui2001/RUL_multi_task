import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch


class MultiTaskDataset(Dataset):
    def __init__(self, features, soh, rul):
        self.features = torch.tensor(features, dtype=torch.float32).unsqueeze(-1)
        self.soh = torch.tensor(soh, dtype=torch.float32)
        self.rul = torch.tensor(rul, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.soh[idx], self.rul[idx]


def create_sequences(df, lookback_len, pred_len):
    X, Y_soh, Y_rul = [], [], []
    for bat, group in df.groupby('battery'):
        # 查找EOL
        eol_cycle = group[group['rul'] == 0].iloc[0]['cycle']

        # 计算可以生成的样本数量，考虑序列长度小于 eol_cycle + pred_len 的情况
        num_samples = min(eol_cycle, len(group) - pred_len) - lookback_len + 1
        for i in range(num_samples):
            X.append(group.iloc[i:(i + lookback_len)]['soh'].to_numpy())
            Y_soh.append(group.iloc[(i + lookback_len):(i + lookback_len + pred_len)]['soh'].to_numpy())
            Y_rul.append(group.iloc[i + lookback_len - 1]['rul'])

    return np.array(X), np.array(Y_soh), np.array(Y_rul)


def create_dataloader(df, lookback_len, pred_len, batch_size, shuffle):
    X, Y_soh, Y_rul = create_sequences(df, lookback_len, pred_len)
    dataset = MultiTaskDataset(X, Y_soh, Y_rul)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_data(df, val_bat, test_bat, lookback_len, pred_len, batch_size):
    train_df = df[~df['battery'].isin([val_bat, test_bat])]
    val_df = df[df['battery'] == val_bat]
    test_df = df[df['battery'] == test_bat]
    train_loader = create_dataloader(train_df, lookback_len, pred_len, batch_size, shuffle=True)
    val_loader = create_dataloader(val_df, lookback_len, pred_len, batch_size, shuffle=False)
    test_loader = create_dataloader(test_df, lookback_len, pred_len, batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
