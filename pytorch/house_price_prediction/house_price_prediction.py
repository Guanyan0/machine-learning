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
train_labels = torch.tensor(train_data.iloc[:, 0].astype(float).values).float()
test_features = torch.tensor(test_data.iloc[:, 1:].astype(float).values).float()
test_labels = torch.tensor(test_data.iloc[:, 0].astype(float).values).float()
loss = nn.MSELoss()
in_features = train_features.shape[1]
net = nn.Sequential(
    nn.Linear(in_features, 1),
)


def load_array(data_arrays, batch_size, is_train=True):
    from torch.utils import data
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 80
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
train_iter = load_array((train_features, train_labels), batch_size)
num_epochs = 30
l_train_history = []
l_test_history = []
for epoch in range(num_epochs):
    l_train = loss(net(train_features), train_labels.reshape(-1, 1))
    l_test = loss(net(test_features), test_labels.reshape(-1, 1))
    l_train_history.append(l_train.item())
    l_test_history.append(l_test.item())

    for X, y in train_iter:
        y = y.reshape(-1, 1)
        loss_f = loss(net(X), y)
        optimizer.zero_grad()
        loss_f.backward()
        optimizer.step()

    scheduler.step()
    print('l_train:', l_train, 'l_test:', l_test)

# 绘制损失曲线
plt.plot(l_train_history, label='Training loss')
plt.plot(l_test_history, label='Test loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Test Loss Over Epochs')
plt.show()
