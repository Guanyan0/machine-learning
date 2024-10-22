import matplotlib.pyplot as plt
from IPython.display import display, clear_output

class Animator:
    """For plotting data in animation."""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        """定义动画器的参数"""
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        plt.ion()  # 开启交互式模式
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: self.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def set_axes(self, ax, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        """设置坐标轴"""
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        if legend:
            ax.legend(legend)

    def add(self, x, y):
        """添加多个数据点到图形中"""
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        plt.show()  # 显示图形
        plt.pause(0.1)  # 暂停以更新图形

# 使用示例
animator = Animator(xlabel='Epoch', ylabel='Accuracy', legend=['Train Acc', 'Test Acc'])

# 模拟训练过程
num_epochs = 10  # 设置训练周期
for epoch in range(num_epochs):
    train_acc = 0.1 * (epoch + 1)  # 模拟训练准确率
    test_acc = 0.1 * (epoch + 1) + 0.05  # 模拟测试准确率
    animator.add(epoch + 1, (train_acc, test_acc))  # 添加数据点
