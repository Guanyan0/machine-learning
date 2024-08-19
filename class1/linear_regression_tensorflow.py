import numpy as np
import tensorflow as tf

X = np.array([[0., 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.float32)
y = np.array([2, 4, 6], dtype=np.float32)
W = tf.Variable(np.array([[1], [1], [1]], dtype=np.float32))
b = tf.Variable(0.0, dtype=np.float32)
iterations = 100
alpha_ = .01

# fwb=tf.linalg.matmul(X,W)+b
# cost_f=(fwb-y.reshape(-1,1))**2/2/len(y)
# print(fwb)
# print(cost_f)
for iter in range(iterations):
    with tf.GradientTape() as tape:
        fwb = tf.linalg.matmul(X, W) + b
        cost_f = tf.reduce_mean((fwb - y.reshape(-1, 1)) ** 2) / 2  # 使用 reduce_mean 计算平均值
    dj_dw, dj_db = tape.gradient(cost_f, [W, b])
    W.assign_sub(alpha_ * dj_dw)  # 使用 assign_sub 减少更新量
    b.assign_sub(alpha_ * dj_db)
    print(iter, '---', cost_f.numpy())

print(W.value(),'\n',b.value())
