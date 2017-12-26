# coding:utf-8

import os
import time

from util import eval_RMSE
import math
import numpy as np
from text_analysis.cnn_model import CNN
from torch.autograd import Variable
import torch

'''
尚未解决的问题：
1、word_embedding
2、batch
3、give_item_weight到底是做什么用的
4、模型中seed的用法
5、一些参数的设置
6、添加cuda参数来控制是否使用GPU
'''


def ConvMF(res_dir, train_user, train_item, valid_user, test_user,
           R, CNN_X, vocab_size, if_cuda, init_W=None, give_item_weight=True,
           max_iter=50, lambda_u=1, lambda_v=100, dimension=50,
           dropout_rate=0.2, emb_dim=200, max_len=300, num_kernel_per_ws=100):
    # explicit setting
    a = 1
    b = 0

    num_user = R.shape[0]
    num_item = R.shape[1]
    PREV_LOSS = 1e-50
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    f1 = open(res_dir + '/state.log', 'w')
    # state.log record

    Train_R_I = train_user[1]  # 6040
    Train_R_J = train_item[1]  # 3544
    Test_R = test_user[1]
    Valid_R = valid_user[1]

    if give_item_weight is True:
        item_weight = np.array([math.sqrt(len(i))
                                for i in Train_R_J], dtype=float)
        item_weight *= (float(num_item) / item_weight.sum())
    else:
        item_weight = np.ones(num_item, dtype=float)

    pre_val_eval = 1e10
    best_tr_eval, best_val_eval, best_te_eval = 1e10, 1e10, 1e10

    # dimension: 用户和物品的隐特征维数
    # emb_dim: 词向量的维数
    # if_cuda: 是否用GPU训练CNN
    cnn_module = CNN(dimension, vocab_size, dropout_rate,
                     emb_dim, max_len, num_kernel_per_ws, if_cuda, init_W)

    # 返回CNN的output
    # size of V is (num_item, dimension)
    if if_cuda:
        cnn_module = cnn_module.cuda()
    theta = cnn_module.get_projection_layer(CNN_X)
    U = np.random.uniform(size=(num_user, dimension))
    V = theta

    endure_count = 5
    count = 0
    # max_iter is 50
    for iteration in range(max_iter):
        loss = 0
        tic = time.time()
        print("%d iteration\t(patience: %d)" % (iteration, count))

        VV = b * (V.T.dot(V)) + lambda_u * np.eye(dimension)
        sub_loss = np.zeros(num_user)

        for i in range(num_user):
            idx_item = train_user[0][i]
            V_i = V[idx_item]
            R_i = Train_R_I[i]
            A = VV + (a - b) * (V_i.T.dot(V_i))
            B = (a * V_i * np.tile(R_i, (dimension, 1)).T).sum(0)

            U[i] = np.linalg.solve(A, B)

            sub_loss[i] = -0.5 * lambda_u * np.dot(U[i], U[i])

        loss = loss + np.sum(sub_loss)

        sub_loss = np.zeros(num_item)
        UU = b * (U.T.dot(U))
        for j in range(num_item):
            idx_user = train_item[0][j]
            U_j = U[idx_user]
            R_j = Train_R_J[j]

            tmp_A = UU + (a - b) * (U_j.T.dot(U_j))
            A = tmp_A + lambda_v * item_weight[j] * np.eye(dimension)
            B = (a * U_j * np.tile(R_j, (dimension, 1)).T
                 ).sum(0) + lambda_v * item_weight[j] * theta[j]
            V[j] = np.linalg.solve(A, B)

            sub_loss[j] = -0.5 * np.square(R_j * a).sum()
            sub_loss[j] = sub_loss[j] + a * np.sum((U_j.dot(V[j])) * R_j)
            sub_loss[j] = sub_loss[j] - 0.5 * np.dot(V[j].dot(tmp_A), V[j])

        loss = loss + np.sum(sub_loss)

        # 用V训练CNN模型，更新V
        cnn_module.train(CNN_X, V)
        theta = cnn_module.get_projection_layer(CNN_X)

        # 这部分添加计算CNN模型的损失
        # cnn_loss = history.history['loss'][-1]

        # loss -= 0.5 * lambda_v * cnn_loss * num_item

        tr_eval = eval_RMSE(Train_R_I, U, V, train_user[0])
        val_eval = eval_RMSE(Valid_R, U, V, valid_user[0])
        te_eval = eval_RMSE(Test_R, U, V, test_user[0])

        # 计算一次迭代的时间
        toc = time.time()
        elapsed = toc - tic

        # 计算Loss下降率
        converge = abs((loss - PREV_LOSS) / PREV_LOSS)

        # 存储效果最好的模型参数
        if val_eval < pre_val_eval:
            torch.save(cnn_module, res_dir+'CNN_model.pt')
            best_tr_eval, best_val_eval, best_te_eval = tr_eval, val_eval, te_eval
            np.savetxt(res_dir + '/U.dat', U)
            np.savetxt(res_dir + '/V.dat', V)
            np.savetxt(res_dir + '/theta.dat', theta)
        else:
            count += 1

        pre_val_eval = val_eval

        print("Elpased: %.4fs Converge: %.6f Train: %.5f Valid: %.5f Test: %.5f" % (
            elapsed, converge, tr_eval, val_eval, te_eval))
        f1.write("Elpased: %.4fs Converge: %.6f Train: %.5f Valid: %.5f Test: %.5f\n" % (
            elapsed, converge, tr_eval, val_eval, te_eval))

        # 超过五次则退出迭代训练
        if count == endure_count:
            print("\n\nBest Model: Train: %.5f Valid: %.5f Test: %.5f" % (
                best_tr_eval, best_val_eval, best_te_eval))
            f1.write("\n\nBest Model: Train: %.5f Valid: %.5f Test: %.5f\n" % (
                best_tr_eval, best_val_eval, best_te_eval))
            break

        PREV_LOSS = loss

    f1.close()
