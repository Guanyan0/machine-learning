import random

import matplotlib.pyplot as plt
import torch
from d2l import torch as d2l


def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)


# 图像
# d2l.set_figsize()
# d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1);
# plt.show()

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


batch_size = 10

# for X, y in data_iter(batch_size, features, labels):
#     print(X, '\n', y)
#     break

w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
print('initial w:', w, ' b:', b)


def linreg(X, w, b):
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def small_batch_GD(params, lr, batch_size):  # 实现1
    for param in params:
        param.data -= lr * param.grad / batch_size
        param.grad.zero_()


def sgd(params, lr, batch_size):  # 实现2
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


lr = 0.03
num_epoch = 3
net = linreg
loss = squared_loss

for epoch in range(num_epoch):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        # small_batch_GD([w,b],lr,batch_size)
        sgd([w, b], lr, batch_size)
    # with torch.no_grad():     # 实现1
    #     train_l = loss(net(features, w, b), labels)
    #     print(f'epoch {epoch + 1}, loss(MSE) {float(train_l.mean()):f}')
    train_l = loss(net(features, w, b), labels)  # 实现2
    print('epoch', epoch + 1, ' loss(MSE):', train_l.mean(), ' w:', w, ' b:', b)

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
