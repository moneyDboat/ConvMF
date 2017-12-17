# coding:utf-8

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class CNN(nn.Module):
    batch_size = 128
    # More than this epoch cause easily over-fitting on our data sets
    nb_epoch = 5

    def __init__(self, output_dimesion, vocab_size, dropout_rate, emb_dim, max_len, n_filters, init_W=None):
        # number_filters
        super(CNN, self).__init__()

        self.max_len = max_len
        max_features = vocab_size
        vanila_dimension = 200  # 倒数第二层的节点数
        projection_dimension = output_dimesion  # 输出层的节点数
        self.qual_conv_set = {}

        '''Embedding Layer'''
        if init_W is None:
            # 先尝试使用embedding随机赋值
            self.embedding = nn.Embedding(max_features, emb_dim)

        self.conv1 = nn.Sequential(
            # 卷积层的激活函数
            nn.Conv2d(1, n_filters, kernel_size=(3, emb_dim)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(max_len - 3 + 1, 1))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, n_filters, kernel_size=(4, emb_dim)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(max_len - 4 + 1, 1))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(1, n_filters, kernel_size=(5, emb_dim)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(max_len - 5 + 1, 1))
        )

        '''Dropout Layer'''
        # layer = Dense(vanila_dimension, activation='tanh')(flatten_layer)
        # layer = Dropout(dropout_rate)(layer)
        self.layer = nn.Linear(300, vanila_dimension)
        self.dropout = nn.Dropout(dropout_rate)

        '''Projection Layer & Output Layer'''
        # output_layer = Dense(projection_dimension, activation='tanh')(layer)
        self.output_layer = nn.Linear(vanila_dimension, projection_dimension)

    def forward(self, input):
        embeds = self.embedding(input)
        # concatenate the tensors
        x = self.conv_1(embeds)
        y = self.conv_2(embeds)
        z = self.conv_3(embeds)
        flatten = torch.cat((x, view(-1), y.view(-1), z.view(-1)))

        out = F.tanh(self.layer(flatten))
        out = self.dropout(out)
        out = F.tanh(self.output_layer(out))


cnn = CNN(50, 8000, 0.5, 50, 150, 100)

