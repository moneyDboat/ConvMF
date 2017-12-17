# coding:utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable


class CNN(nn.Module):
    batch_size = 128
    # More than this epoch cause easily over-fitting on our data sets
    nb_epoch = 5

    def __init__(self, output_dimesion, vocab_size, dropout_rate, emb_dim, max_len, nb_filters, init_W=None):
        super(CNN, self).__init__()

        self.max_len = max_len
        max_features = vocab_size
        vanila_dimension = 200
        projection_dimension = output_dimesion
        self.qual_conv_set = {}

        '''Embedding Layer'''
        if init_W is None:
            self.embedding = nn.Embedding(max_features, emb_dim)

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, nb_filters, kernel_size=(3, emb_dim)),
            nn.MaxPool2d(kernel_size=(max_len - 3 + 1, 1))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, nb_filters, kernel_size=(4, emb_dim)),
            nn.MaxPool2d(kernel_size=(max_len - 4 + 1, 1))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(1, nb_filters, kernel_size=(5, emb_dim)),
            nn.MaxPool2d(kernel_size=(max_len - 5 + 1, 1))
        )


        self.conv_1 = nn.Conv2d(1, nb_filters, kernel_size=(3, emb_dim))
        self.conv_2 = nn.Conv2d(1, nb_filters, kernel_size=(4, emb_dim))
        self.conv_3 = nn.Conv2d(1, nb_filters, kernel_size=(5, emb_dim))

    def forward(self, *input):
        pass
