import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, hidden_dim, seq_len, ker_num, dropout, num_class, vocab_size=21128):
        super(TextCNN, self).__init__()
        # 论文原文中
        # hidden_dim = 128
        # ker_num = 100
        # dropout = 0.5
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.drop_input = nn.Dropout(p=0.5)

        # torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0）
        # in_channels：输入信号的通道，由词向量的维度决定
        # out_channels：输出的通道数
        # kernel_size：卷积核的尺寸，实际为embed_dim*kernel_size
        # stride：步长， padding：补0的层数

        self.conv0 = nn.Conv1d(in_channels=hidden_dim, out_channels=ker_num, kernel_size=3, padding=1)
        self.conv1 = nn.Conv1d(in_channels=hidden_dim, out_channels=ker_num, kernel_size=4, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=ker_num, kernel_size=5, padding=2)
        # self.conv3 = nn.Conv1d(in_channels=seq_len, out_channels=ker_num, kernel_size=5)
        # self.conv4 = nn.Conv1d(in_channels=seq_len, out_channels=ker_num, kernel_size=6)

        self.dropout = nn.Dropout(p=dropout)

        self.fc = nn.Linear(ker_num*3, num_class)

    def forward(self, x, attention_mask):

        embed = self.embed(x).permute(dims=[0, 2, 1])
        embed = self.drop_input(embed)
        x0 = torch.sigmoid(self.conv0(embed))
        x1 = torch.tanh(self.conv1(embed))
        x2 = torch.relu(self.conv2(embed))
        # x3 = torch.sigmoid(self.conv3(embed))
        # x4 = torch.sigmoid(self.conv4(embed))

        # x1 = self.dropout(x1)
        # x2 = self.dropout(x2)
        # x3 = self.dropout(x3)

        x_mp0 = F.max_pool1d(x0, x0.size(2)).squeeze(2)
        x_mp1 = F.max_pool1d(x1, x1.size(2)).squeeze(2)
        x_mp2 = F.max_pool1d(x2, x2.size(2)).squeeze(2)
        # x_mp3 = F.max_pool1d(x3, x3.size(2)).squeeze(2)
        # x_mp4 = F.max_pool1d(x4, x4.size(2)).squeeze(2)

        x_cat = torch.cat((x_mp0, x_mp1, x_mp2), dim=1)
        x_cat = self.dropout(x_cat)
        output = self.fc(x_cat)

        return output


class slicenet(nn.Module):
    def __init__(self):
        super(slicenet, self).__init__()


    def forward(self):
        pass



import torch
import torch.nn as nn


class SepConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size, stride, padding, groups=in_channel)
        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, input):
        x = self.conv1(input)
        print(self.conv1.weight.data.shape)
        x = self.conv2(x)
        print(self.conv2.weight.data.shape)
        return x


if __name__ == "__main__":
    input = torch.rand(2, 16, 64)
    a = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=0, groups=16)
    print(a.weight.data.shape)
    b = a(input)
    print(b.shape)
