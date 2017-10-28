import numpy as np
import PolynomialAndData.polynomial as poly

def GradientDescentRegular(x, y, w_init):
    #添加正则项的梯度下降算法
    lamda = 0.001  # 设置正则系数
    w = np.mat(w_init.copy()).T#将数据转化为矩阵
    Y = np.mat(y).T
    step = 0.05 #设置布长为0.05
    X = np.mat(np.zeros((x.size, w_init.size)))#范德蒙德矩阵的计算
    for i in range(x.size):
        for j in range(w_init.size):
            X[i, j] = pow(x[i], j)

    loss_function = 1.0#初始设置损失函数值
    iteratornNum = 0#迭代次数极为0
    while (loss_function > 0.001 and iteratornNum < 200000):
        #迭代终止条件为loss_function > 0.001 and iteratornNum < 200000
        iteratornNum = iteratornNum + 1
        w -= step * (np.dot(X.T, (np.dot(X, w) - Y)) + lamda * w)#更新w，并设置惩罚系数为0.001
        loss_function = 0.0
        for i in range(x.size):
            loss_function += pow(poly.polynomial(x[i], np.array(w.T)[0]) - y[i], 2)
        loss_function /= (2 * x.size)#计算损失函数值
    W = np.zeros(w_init.size)
    for i in range(w_init.size):
        W[i] = w[i, 0]
    print("loss_function:", loss_function)
    print("The iteratorNum times is:", iteratornNum)
    return W


def GradientDescent(x, y, w_init):
    # 正宗梯度下降算法
    w = np.mat(w_init.copy()).T
    Y = np.mat(y).T
    step = 0.05
    X = np.mat(np.zeros((x.size, w_init.size)))#范德蒙德矩阵的计算
    for i in range(x.size):
        for j in range(w_init.size):
            X[i, j] = pow(x[i], j)
    loss_function = 1.0  # 初始设置损失函数值
    iteratornNum = 0  # 迭代次数极为0
    while (loss_function > 0.001 and iteratornNum < 200000):
        # 迭代终止条件为loss_function > 0.001 and iteratornNum < 200000
        iteratornNum = iteratornNum + 1
        w -= step * np.dot(X.T, (np.dot(X, w) - Y))  # 更新w，并设置惩罚系数为0.001
        loss_function = 0.0
        for i in range(x.size):
            loss_function += pow(poly.polynomial(x[i], np.array(w.T)[0]) - y[i], 2)
        loss_function /= (2 * x.size)  # 计算损失函数值
    W = np.zeros(w_init.size)
    for i in range(w_init.size):
        W[i] = w[i, 0]
    print("loss_function:", loss_function)
    print("The iteratorNum times is:", iteratornNum)
    return W
