import torch
import torch.nn as nn

X = torch.normal(0, 1, size=(3, 5, 5))
# print(X)
K = torch.rand(size=(3, 3, 3))


def corr2d(X, K):
    m, n = K.shape
    Y = torch.zeros((X.shape[0] - m + 1, X.shape[1] - n + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + m, j:j + n] * K).sum()
    return Y


def corr2d_multi_in(X, K):
    return sum(corr2d(x,k) for x,k in zip(X,K))
print(X.shape,K.shape)
out=corr2d_multi_in(X, K)
print(out)

def corr2d_multi_in_out(X, K):
    # 迭代“K”的第0个维度，每次都对输入“X”执行互相关运算。
    # 最后将所有结果都叠加在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)
K = torch.stack((K, K + 1, K + 2), 0)
print(K.shape)
out_3=corr2d_multi_in_out(X, K)
print(out_3)
