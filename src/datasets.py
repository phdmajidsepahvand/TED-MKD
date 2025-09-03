
import torch
class CrossWindowECGDataset(Dataset):
    def __init__(self, X, y, windows_per_sample=3):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.wps = windows_per_sample
        self.valid_len = len(self.X) - self.wps + 1

    def __len__(self):
        return self.valid_len

    def __getitem__(self, idx):
        x_group = self.X[idx:idx+self.wps]  # [wps, 3000]
        y_label = self.y[idx + self.wps - 1]

        if torch.isnan(x_group).any():
            x_group = torch.nan_to_num(x_group, nan=0.0)

        return x_group, y_label

    
class BaselineDataset(Dataset):
    def __init__(self, X, y):
        self.X = X[:, 0, :]
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
        
