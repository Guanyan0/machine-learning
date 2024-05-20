import numpy as np
import matplotlib.pyplot as plt

# different start number have different result
# X = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
X = np.array([[0., 100, 2], [3, 400, 5], [6, 700, 8]])
# y = np.array([2, 4, 6.1])
y = np.array([2, 4, 6])
# w_start = np.array([1, 1, 1])
# w_start = np.array([0, 0, 0])
# w_start = np.array([100, 0, 0])
w_start = np.zeros(3)
b_start = 0
a = 0.1
cost = 999.


def cal_cost(y, X, w, b):
    cost_c = 0
    for i in range(len(y)):
        cost_c += (y[i] - np.dot(w, X[i]) - b) ** 2 / 2 / len(y)
    return cost_c


# w = w_start
# b = b_start
# cost = cal_cost(y, X, w, b)
# print(type(cost))


def compute_gradient(y, X, w, b):
    m, n = X.shape  # (number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i, j]
        dj_db = dj_db + err
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


def draw_gradient_descent_curve(results, title='Relationship between Cost and Count'):
    cost_values = [row[-1] for row in results]
    count_values = [row[-2] for row in results]

    # 绘制连线的散点图
    plt.figure(figsize=(10, 6))
    plt.plot(count_values, cost_values, marker='o', linestyle='-', color='blue')
    plt.xlabel('Iteration Count')
    plt.ylabel('Cost')
    plt.title(title)
    plt.grid(True)
    plt.show()


def draw_target_linear_regression(results, y, X):
    # w = results[len(results) - 1][1]
    # print(w)
    # b = results[len(results) - 1][2]
    # plt.figure(1)
    # hx = np.linspace(0, X.max(), 1000)
    # h = np.dot(w, hx)+b
    # plt.plot(hx,h)
    # # print(X.max())



    # 原始数据
    X = np.array([[0., 100, 2], [3, 400, 5], [6, 700, 8]])
    y = np.array([2, 4, 6])

    # 计算线性回归系数
    A = np.hstack([X, np.ones((X.shape[0], 1))])
    w = np.linalg.lstsq(A, y, rcond=None)[0]

    # 绘制数据点
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], y, color='r', marker='o', label='Original Data')

    # 计算回归面上的点
    x1 = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    x2 = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)
    x1, x2 = np.meshgrid(x1, x2)
    y_pred = w[0] * x1 + w[1] * x2 + w[2]

    # 绘制回归面
    ax.plot_surface(x1, x2, y_pred, alpha=0.5, cmap='viridis', label='Regression Plane')

    # 设置图像标题和轴标签
    ax.set_title('Linear Regression in 3D')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')

    # 添加图例
    ax.legend()

    # 显示图形
    plt.show()


def mul_gradient_descent_by_error_rate(y, X, w_start, b_start, error_range):
    results = []
    # error_range = np.array([1,0.1])
    for i in range(error_range.shape[0]):
        w = w_start
        b = b_start
        a = 0.1
        cost = 999.
        iteration_count = 0
        while cost > error_range[i] and iteration_count < 100000:
            w_i, b_i = compute_gradient(y, X, w, b)
            w = w - a * w_i
            b = b - a * b_i
            cost_1 = cal_cost(y, X, w, b)
            if cost_1 > cost:
                a = a * 0.5
            cost = cost_1
            iteration_count += 1
        # print(error_range[i], w, b, iteration_count, cost)
        results.append((error_range[i], w, b, iteration_count, cost))

    # print(results)
    return results


def mul_gradient_descent_by_times(y, X, w_start, b_start, times=10000, point_num=25):
    results = []
    w = w_start
    b = b_start
    a = 0.1
    cost = 999.
    iteration_count = 0
    while iteration_count < times:
        w_i, b_i = compute_gradient(y, X, w, b)
        w = w - a * w_i
        b = b - a * b_i
        cost_1 = cal_cost(y, X, w, b)
        if cost_1 > cost:
            a = a * 0.5
        cost = cost_1
        iteration_count += 1
        if np.mod(iteration_count, times / point_num) == 0:
            results.append((iteration_count, w, b, iteration_count, cost))

    return results


def mean_normalization(X):
    X_out = X.copy()
    m, n = X_out.shape
    for i in range(n):
        mean = np.mean(X_out[i])
        x_max = np.max(X_out[i])
        x_min = np.min(X_out[i])
        for j in range(m):
            X_out[j][i] = (X_out[j][i] - mean) / (x_max - x_min)
    return X_out


X_norm = mean_normalization(X)
error_range = np.array([0.1, 0.01, 0.001, 0.000001, 1.0e-10])
# rs = mul_gradient_descent_error_rate(y, X_norm, w_start, b_start, error_range)
rs = mul_gradient_descent_by_times(y, X_norm, w_start, b_start, 500)
# print(rs)

draw_gradient_descent_curve(rs, 'after normalization')
draw_target_linear_regression(rs, y, X_norm)
