import os

import pandas as pd

os.makedirs(os.path.join('../..', 'pytorch', 'data'), exist_ok=True)
datafile = os.path.join('../..', 'pytorch', 'data', 'house.csv')
with open(datafile, 'w') as f:
    f.write('num_room,alley,price\n')
    f.write('NA,Pave,127500\n')
    f.write('2,Star,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

data = pd.read_csv('data/house.csv')
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
print('before:\n', inputs)

# 数据预处理
inputs = inputs.fillna(inputs.mean(numeric_only=True))  # fill null with average value
inputs = pd.get_dummies(inputs,dummy_na=True)  # one-hot encoding
print('after:\n', pd.concat([inputs, outputs], axis=1))  # 按列合并
