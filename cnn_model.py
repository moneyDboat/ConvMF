# coding:utf-8

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class CNN(nn.Module):
    def __init__(self, output_dimesion, vocab_size, dropout_rate, emb_dim, max_len, n_filters, init_W=None):
        # number_filters
        print(type(self))
        super(CNN, self).__init__()

        self.max_len = max_len
        self.emb_dim = emb_dim
        vanila_dimension = 200  # 倒数第二层的节点数
        projection_dimension = output_dimesion  # 输出层的节点数
        self.qual_conv_set = {}

        '''Embedding Layer'''
        if init_W is None:
            # 先尝试使用embedding随机赋值
            self.embedding = nn.Embedding(vocab_size, emb_dim)

        self.conv1 = nn.Sequential(
            # 卷积层的激活函数
            # 将embedding dimension看做channels数
            nn.Conv1d(emb_dim, n_filters, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=max_len - 3 + 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(emb_dim, n_filters, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=max_len - 4 + 1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(emb_dim, n_filters, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=max_len - 5 + 1)
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
        # 进入卷积层前需要将Tensor第二个维度变成emb_dim，作为卷积的通道数
        embeds = embeds.view([len(embeds), self.emb_dim, -1])
        # concatenate the tensors
        x = self.conv1(embeds)
        y = self.conv2(embeds)
        z = self.conv3(embeds)
        flatten = torch.cat((x.view(-1), y.view(-1), z.view(-1)), 1)

        out = F.tanh(self.layer(flatten))
        out = self.dropout(out)
        out = F.tanh(self.output_layer(out))

    def train(self, X_train, V, item_weight, seed):
        pass

    # 获取CNN模型的输出

    # def train(self, X_train, V, item_weight, seed):
    #     # X_train is CNN_X
    #     X_train = sequence.pad_sequences(X_train, maxlen=self.max_len)
    #     np.random.seed(seed)
    #     X_train = np.random.permutation(X_train)
    #     np.random.seed(seed)
    #     V = np.random.permutation(V)
    #     np.random.seed(seed)
    #     item_weight = np.random.permutation(item_weight)
    #
    #     print("Train...CNN module")
    #     history = self.model.fit(X_train, V, verbose=0, batch_size=self.batch_size,
    #                              epochs=self.nb_epoch, sample_weight=item_weight)
    #
    #     # cnn_loss_his = history.history['loss']
    #     # cmp_cnn_loss = sorted(cnn_loss_his)[::-1]
    #     # if cnn_loss_his != cmp_cnn_loss:
    #     #     self.nb_epoch = 1
    #     return history
    #
    # def get_projection_layer(self, X_train):
    #     X_train = sequence.pad_sequences(X_train, maxlen=self.max_len)
    #     Y = self.model.predict(X_train, batch_size=len(X_train))
    #     return Y
