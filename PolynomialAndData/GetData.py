import numpy as np
import scipy as sp

def GaussianNoise(size=10):
    #生成满足高斯分布噪音，满足均值为0
    noise = np.random.normal(0,0.03,size=size)
    noise = noise - noise.sum(axis=0)/size
    return noise

def sin(size =10):
    #生成具有高斯分布噪音的数据点
    x = np.random.rand(size)
    x.sort(axis=0)
    y = np.sin(2*np.pi*x)
    y = y + GaussianNoise(size)
    return x, y

