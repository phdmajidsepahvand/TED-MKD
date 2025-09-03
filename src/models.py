import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class CrossWindowAttentionModel(nn.Module):
    def __init__(self, window_size=3000, windows_per_sample=3, embed_dim=128, num_heads=4):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(window_size, embed_dim),
            nn.ReLU(),
        )

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: [B, W, 3000]
        x = self.embed(x)  # → [B, W, D]
        attn_output, _ = self.attn(x, x, x)  # self-attention → [B, W, D]
        x_pooled = attn_output.mean(dim=1)  # → [B, D]
        out = self.classifier(x_pooled)     # → [B, 1]
        return out

class CNNTeacherModel(nn.Module):
    def __init__(self):
        super(CNNTeacherModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return torch.sigmoid(x)



class StudentModel(nn.Module):
    def __init__(self, input_size=3000*3):
        super(StudentModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # [B, 3, 3000] → [B, 9000]
        return torch.sigmoid(self.net(x))




class BaselineECGModel(nn.Module):
    def __init__(self, input_size=3000):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

