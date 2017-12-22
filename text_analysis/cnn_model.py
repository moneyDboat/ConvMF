# coding:utf-8

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.autograd import Variable


class CNN(nn.Module):
    batch_size = 128
    # More than this epoch cause easily over-fitting on our data sets
    nb_epoch = 5

    def __init__(self, output_dimesion, vocab_size, dropout_rate, emb_dim, max_len, n_filters, init_W=None):
        # n_filter为卷积核个数
        super(CNN, self).__init__()

        self.max_len = max_len
        self.emb_dim = emb_dim
        vanila_dimension = 200  # 倒数第二层的节点数
        projection_dimension = output_dimesion  # 输出层的节点数
        self.qual_conv_set = {}

        '''Embedding Layer'''
        # if init_W is None:
        #     # 最后一个索引为填充的标记文本
        #     # 先尝试使用随机生成的词向量值
        self.embedding = nn.Embedding(vocab_size + 1, emb_dim)

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

    def forward(self, inputs):
        size = len(inputs)
        embeds = self.embedding(inputs)

        # 进入卷积层前需要将Tensor第二个维度变成emb_dim，作为卷积的通道数
        embeds = embeds.view([len(embeds), self.emb_dim, -1])
        # concatenate the tensors
        x = self.conv1(embeds)
        y = self.conv2(embeds)
        z = self.conv3(embeds)
        flatten = torch.cat((x.view(size, -1), y.view(size, -1), z.view(size, -1)), 1)

        out = F.tanh(self.layer(flatten))
        out = self.dropout(out)
        out = F.tanh(self.output_layer(out))

        return out

    def train(self, X_train, V):

        # learning rate暂时定为0.001
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        for epoch in range(1, self.nb_epoch + 1):

            print('<---epoch' + str(epoch))
            n_batch = len(X_train) // self.batch_size

            # 这里会漏掉一些训练集，先这样写
            for i in range(n_batch):
                begin_idx, end_idx = i * self.batch_size, (i + 1) * self.batch_size
                feature = X_train[begin_idx:end_idx][...]
                target = V[begin_idx:end_idx][...]

                feature = Variable(torch.from_numpy(feature.astype('int64')).long())
                target = Variable(torch.from_numpy(target))
                feature, target = feature.cuda(), target.cuda()

                optimizer.zero_grad()
                logit = self(feature)

                loss = F.mse_loss(logit, target)
                loss.backward()
                optimizer.step()

    def get_projection_layer(self, X_train):
        inputs = Variable(torch.from_numpy(X_train.astype('int64')).long())
        inputs = inputs.cuda()
        outputs = self(inputs)
        return outputs.cpu().data.numpy()


    # 获取CNN模型的输出

    # def train(self, X_train, V, item_weight, seed):
    #     # X_train is CNN_X
    #     X_train = sequence.pad_sequences(X_train, maxlen=self.max_len)
    #     np.random.seed(seed)
    #     X_train = np.random.permutation(X_train)
    #     np.random.seed(seed)
    #     V = np.random.permutation(V)ojecti
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
