import torch
import torch.nn as nn
import torch.nn.functional as F


class TextRCNN(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout, num_class, vob_size=21128):
        super(TextRCNN, self).__init__()

        self.embed = nn.Embedding(vob_size, embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim,
                            batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(embed_dim + hidden_dim, 128)
        self.fc2 = nn.Linear(128, num_class)

    def forward(self, x, attention_mask):
        x_embed = self.embed(x)
        output, (h, c) = self.lstm(x_embed)
        out = torch.cat((x_embed, output), dim=2)
        # 论文原文这里应该是用tanh函数
        out = F.relu(self.fc1(out))
        out = out.permute(dims=[0, 2, 1])
        out = F.max_pool1d(out, out.size(-1)).squeeze(-1)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
