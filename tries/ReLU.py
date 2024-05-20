import matplotlib.pyplot as plt
import numpy as np


def ReLU(x_start, x_end):
    x_out = np.arange(x_start, x_end)
    y_out = np.maximum(0, x_out)
    print(y_out)

    plt.plot(x_out, y_out)
    plt.show()

    return x_out, y_out


ReLU(-100, 100)
