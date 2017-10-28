import numpy as np
import matplotlib.pyplot as plot
import PolynomialAndData.GetData as gd

import method.ConjungateGradient as CG
import PolynomialAndData.polynomial as poly

size = 10
power = 5
x,y=gd.sin(size)


x_points=np.linspace(0,1,num=1000)
w_init=np.random.randn(power)
print("x",x)
print("y",y)
print("w_init:",w_init)


plot.title("Conjungate Gradient")
plot.plot(x_points,np.sin(2*np.pi*x_points),label='$sin(2*pi*x)$',color="red")

for i in range(size):
    plot.scatter(x[i],y[i],color="green", linewidths=0.01)

w_FR1=CG.ConjungateGradient(x,y,w_init)
print("w_FR1:",w_FR1)
plot.plot(x_points,poly.polynomial(x_points,w_FR1),label= '$CG$',color="green")

w_FR2=CG.ConjungateGradientRegular(x,y,w_init)
print("w_FR2:",w_FR2)
plot.plot(x_points,poly.polynomial(x_points,w_FR2),label='$CG+Regular$',color="blue")

plot.legend()
plot.show()
