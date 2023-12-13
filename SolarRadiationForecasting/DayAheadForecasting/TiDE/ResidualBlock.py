import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.0, layer_norm=True):
        super(ResidualBlock, self).__init__()
        self.layer_norm = layer_norm
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.residual_fc = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        if self.layer_norm:
            self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        residual = self.residual_fc(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(self.fc2(x))
        x = self.norm(x + residual)
        return x
