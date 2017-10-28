import numpy as np
import matplotlib.pyplot as plot
import PolynomialAndData.GetData as gd
import method.LeastSquare as LS
import PolynomialAndData.polynomial as poly

size = 10
power = 10
x,y=gd.sin(size)
x_points=np.linspace(0,1,num=1000)
w_init=np.random.randn(power)
print("x",x)
print("y",y)
print("w_init:",w_init)

plot.title("Least Square")
plot.plot(x_points,np.sin(2*np.pi*x_points),label='$sin(2*pi*x)$',color="red")

for i in range(size):
    plot.scatter(x[i],y[i],color="green", linewidths=0.01)

w_LS1=LS.leastSqure(x,y,w_init)
print("w_LS1:",w_LS1)
plot.plot(x_points,poly.polynomial(x_points,w_LS1),label= '$LS$',color="green")

w_LS2=LS.leastSqureRegular(x,y,w_init)
print("w_LS2:",w_LS2)
plot.plot(x_points,poly.polynomial(x_points,w_LS2),label= '$LS+Regular$',color="blue")

plot.legend()
plot.show()
