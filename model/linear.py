import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepNet(nn.Module):
    def __init__(self, input_size, num_class, dropout, vob_size=5049):
        super(DeepNet, self).__init__()

        self.embed = nn.Embedding(vob_size, input_size)
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_class)
        self.norm1 = nn.LayerNorm(256)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.norm1(x)
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        x = self.norm1(x)
        x = self.dropout(x)
        x = x.permute(dims=[0, 2, 1])
        x = F.max_pool1d(x, x.size(-1)).squeeze(-1)
        x = self.fc3(x)
        return x