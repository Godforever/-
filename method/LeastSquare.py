import numpy as np



def leastSqure(x, y, w_init):
    #最小二乘法
    X = np.mat(np.zeros((x.size, w_init.size)))#范德蒙德矩阵的计算
    for i in range(x.size):
        for j in range(w_init.size):
            X[i, j] = pow(x[i], j)
    w = np.zeros(w_init.size)
    for i in range(w_init.size):
        w[i] = np.dot(np.dot(np.dot(X.T, X).I, X.T), y)[0, i]
        #w等于X的逆矩阵乘以y
    print("w: ", w)
    return w


def leastSqureRegular(x, y, w_init):
    #添加惩罚项的最小二乘法
    lamda = 2  # 设置正则系数
    X = np.mat(np.zeros((x.size, w_init.size)))#范德蒙德矩阵的计算
    for i in range(x.size):
        for j in range(w_init.size):
            X[i, j] = pow(x[i], j)
    w = np.zeros(w_init.size)
    for i in range(w_init.size):
        w[i] = np.dot(np.dot((np.dot(X.T, X) - lamda).I, X.T), y)[0, i]
    print("w: ", w)
    return w


