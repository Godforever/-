import numpy as np

def d1Size(X):
    sum = 0.0
    for i in range(X.size):
        sum += pow(X[i,0],2)
    return sum

def ConjungateGradient(x, y, w_init):
    #不添加正则项的共轭梯度法
    w = np.mat(w_init.copy()).T#实现数据的矩阵化
    Y = np.mat(y.copy()).T
    X = np.mat(np.zeros((x.size, w_init.size)))
    for i in range(x.size):
        for j in range(w_init.size):
            X[i, j] = pow(x[i], j)
    A = np.dot(X.T,X)#正定矩阵
    g = (np.dot(np.dot(X.T,X),w)-np.dot(X.T,Y))
    if(d1Size(g)!=0):
        d = -g
        while(d1Size(g)>0.00001):#计算梯度大小
            step = (np.dot(-g.T,d)/np.dot(np.dot(d.T,A),d))[0,0]
            w += step*d
            g = (np.dot(np.dot(X.T, X), w) - np.dot(X.T, Y))
            speed = (np.dot(np.dot(d.T,A),g)/np.dot(np.dot(d.T,A),d))[0,0]#计算步长
            d = -g+speed*d
    new_w = np.zeros(w_init.size)
    for i in range(w_init.size):
        new_w[i] = w[i,0]
    return new_w

def ConjungateGradientRegular(x,y,w_init):
    # 添加正则项的共轭梯度法
    lamda=4
    w = np.mat(w_init.copy()).T#实现数据的矩阵化
    Y = np.mat(y.copy()).T
    X = np.mat(np.zeros((x.size, w_init.size)))
    for i in range(x.size):
        for j in range(w_init.size):
            X[i, j] = pow(x[i], j)
    E = np.identity(w_init.size)
    A = np.dot(X.T,X) + E*lamda#添加惩罚项的正定矩阵
    g = (np.dot(np.dot(X.T, X), w) - np.dot(X.T, Y))
    if(d1Size(g)!=0):
        d = -g
        while(d1Size(g)>0.00001):#计算梯度大小
            step = (np.dot(-g.T,d)/np.dot(np.dot(d.T,A),d))[0,0]
            w += step*d
            g = (np.dot(np.dot(X.T, X), w) - np.dot(X.T, Y))
            speed = (np.dot(np.dot(d.T,A),g)/np.dot(np.dot(d.T,A),d))[0,0]#计算步长
            d = -g+speed*d
    new_w = np.zeros(w_init.size)
    for i in range(w_init.size):
        new_w[i] = w[i,0]
    return new_w
