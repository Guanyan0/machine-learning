import numpy as np
import time


def print_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 执行函数
        end_time = time.time()  # 记录结束时间
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")  # 打印执行时间
        return result  # 返回函数结果

    return wrapper


@print_execution_time
def matrix_cal(A, B):
    m, n = A.shape
    rs = np.matmul(A, B)
    # rs = A @ B  #  python 3.10后
    return rs


@print_execution_time
def dot_cal(A, B):
    m = A.shape[0]
    n = B.shape[1]
    rs = np.zeros((m, n))
    for j in range(n):
        for i in range(m):
            rs[i][j] = np.dot(A[i, :], B[:, j])
    return rs


# w1 = np.array([[1, 2, 3], [4, 5, 6]])
# X = np.array([1, 2])
# X1 = np.array([[1, 2], [3, 4]])
# X=np.array([[1],[2]])

# rs1=matrix_cal(w1, X1)
# rs2=dot_cal(w1, X1)
# print(rs1,'\n',rs2)

np.random.seed(0)
matrix_A = np.random.randint(1, 11, size=(200, 2000))
matrix_B = np.random.randint(1, 11, size=(2000, 250))

matrix_A1 = np.random.randint(1, 11, size=(20, 20000))
matrix_B1 = np.random.randint(1, 11, size=(20000, 25))

print('结果1：')
rs1 = matrix_cal(matrix_A, matrix_B)
rs2 = dot_cal(matrix_A, matrix_B)

print('结果2：')
rs3 = matrix_cal(matrix_A1, matrix_B1)
rs4 = dot_cal(matrix_A1, matrix_B1)
# 现有两个m1乘n1和m2乘n2的矩阵，在上面两个函数运行中发现，m1、n2同n1的差距越大，点乘方法的效率越高，而m1、n2同n1的差距越小，则矩阵乘法“@”的效率更高
