import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')
# divide the dataset
length = data.shape[0]

# pre-processing
# 注意，在回归问题中，标签y也需要进行标准化
numeric_features = data.dtypes[data.dtypes != 'object'].drop(['yr_renovated']).index
data[numeric_features] = data[numeric_features].apply(
    lambda x: (x - x.mean()) / x.std()
)
data[numeric_features] = data[numeric_features].fillna(0)

data['zipcode'] = data['statezip'].str.extract('(\d{5})').astype(float)
data = pd.get_dummies(data, columns=['city'])
# print(data['statezip'])

# 仅考虑数值特征
data = data.drop(columns=['date', 'street', 'statezip', 'country'])
train_data = data.iloc[:int(0.8 * length)]
test_data = data.iloc[int(0.8 * length):]

print('train_data:', train_data.shape, 'test_data:', test_data.shape)

# print(train_data.dtypes)
train_features = torch.tensor(train_data.iloc[:, 1:].astype(float).values).float()
train_labels = torch.tensor(train_data.iloc[:, 0].astype(float).values).float().reshape(-1, 1)
test_features = torch.tensor(test_data.iloc[:, 1:].astype(float).values).float()
test_labels = torch.tensor(test_data.iloc[:, 0].astype(float).values).float().reshape(-1, 1)
loss = nn.MSELoss()
in_features = train_features.shape[1]
net = nn.Sequential(
    nn.Linear(in_features, 1),
)


train_l_init=loss(net(train_features),train_labels)

def load_array(data_arrays, batch_size, is_train=True):
    from torch.utils import data
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 80
train_iter = load_array((train_features, train_labels), batch_size)


def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(loss(net(train_features), train_labels))
        if test_labels is not None:
            test_ls.append(loss(net(test_features), test_labels))
    return train_ls, test_ls


def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_lost_sum, validation_lost_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        train_loss, test_loss = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_lost_sum += train_loss[-1]
        validation_lost_sum += test_loss[-1]
        print(f'train loss:{train_loss[-1]}\nvalidation loss:{test_loss[-1]}')
    return train_lost_sum / k, validation_lost_sum / k


k, num_epoch, lr, weight_decay = 5, 30, 0.01, 1e-5
train_l, validation_l = k_fold(k, train_features, train_labels, num_epoch, lr, weight_decay, batch_size)
print(f'k fold train loss:{train_l},validation loss:{validation_l}')

train_l_end=loss(net(train_features),train_labels)
test_l=loss(net(test_features),test_labels)
print(f'initial train loss:{train_l_init}\nend train loss:{train_l_end}\ntest set loss:{test_l}')