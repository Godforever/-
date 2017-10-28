import numpy as np
import matplotlib.pyplot as plot
import PolynomialAndData.GetData as gd
import method.GradientDescent as GD
import PolynomialAndData.polynomial as poly

size = 100
power = 10
x,y=gd.sin(size)
x_points=np.linspace(0,1,num=1000)
w_init=np.random.randn(power)
print("x",x)
print("y",y)
print("w_init:",w_init)

plot.title("Gradient Descent")
plot.plot(x_points,np.sin(2*np.pi*x_points),label='$sin(2*pi*x)$',color="red")

for i in range(size):
    plot.scatter(x[i],y[i],color="green", linewidths=0.01)


w_GD1=GD.GradientDescent(x,y,w_init)
print("w_GD1:",w_GD1)
plot.plot(x_points,poly.polynomial(x_points,w_GD1),label= '$GD$',color="blue")

w_GD2=GD.GradientDescentRegular(x,y,w_init)
print("w_GD2:",w_GD2)
plot.plot(x_points,poly.polynomial(x_points,w_GD2),label= '$GD+Regular$',color="green")

plot.legend()
plot.show()
