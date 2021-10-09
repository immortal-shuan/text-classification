import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, tensor


# 参数
class Config:
    # 预处理参数
    window_size = 20  # 20
    # 训练参数
    learning_rate = 0.02  # 0.02
    l2_loss = 0.  # 0.
    val_part = 0.1  # 0.1
    max_epoch = 200  # 200
    stop_epoch = 10  # 10
    optimizer = 'adam'  # adam
    # 模型参数
    dropout = 0.5  # 0.5
    embedding_dim = 200  # 200
    num_nodes = 2294
    num_document = 1000
    num_classes = 2
    # 预训练GloV
    embedding_size = 300  # 300


class TextGCN(nn.Module):
    def __init__(self, A, config):
        super(TextGCN, self).__init__()
        num_nodes = config.num_nodes
        num_document = config.num_document
        embedding_dim = config.embedding_dim
        num_classes = config.num_classes
        dropout = config.dropout
        self.document = num_document
        self.A = nn.Parameter(tensor(A, dtype=torch.float), requires_grad=False)
        self.W0 = nn.Linear(num_nodes, embedding_dim, bias=True)
        self.W1 = nn.Linear(embedding_dim, num_classes, bias=True)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.W0(self.A.mm(x))
        x = F.relu(x, inplace=True)
        x = self.W1(self.A.mm(x))
        x = self.dropout(x)
        x = F.softmax(x[: self.document], dim=1)
        return x


if __name__ == '__main__':
    matrix = np.random.random((Config.num_nodes, Config.num_nodes))

    target = torch.from_numpy(np.random.randint(0, Config.num_classes, Config.num_document))

    model = TextGCN(matrix, Config)

    x = tensor(np.diag(np.ones(Config.num_nodes)), dtype=torch.float)

    y = model(x)

    print(y.size())