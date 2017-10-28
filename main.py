import numpy as np
import matplotlib.pyplot as plot
import PolynomialAndData.GetData as gd
import method.LeastSquare as LS
import method.GradientDescent as GD
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

plot.title("sin function")
plot.plot(x_points,np.sin(2*np.pi*x_points),label='$sin(2*pi*x)$',color="red")

for i in range(size):
    plot.scatter(x[i],y[i],color="green", linewidths=0.01)

w_LS1=LS.leastSqure(x,y,w_init)
print("w_LS1:",w_LS1)
plot.plot(x_points,poly.polynomial(x_points,w_LS1),label= '$LS$',color="green")

w_LS2=LS.leastSqureRegular(x,y,w_init)
print("w_LS2:",w_LS2)
plot.plot(x_points,poly.polynomial(x_points,w_LS2),label= '$LS+Regular$',color="blue")

w_GD1=GD.GradientDescent(x,y,w_init)
print("w_GD1:",w_GD1)
plot.plot(x_points,poly.polynomial(x_points,w_GD1),label= '$GD$',color="yellow")

w_GD2=GD.GradientDescentRegular(x,y,w_init)
print("w_GD2:",w_GD2)
plot.plot(x_points,poly.polynomial(x_points,w_GD2),label= '$GD+Regular$',color="pink")

w_FR1=CG.ConjungateGradient(x,y,w_init)
print("w_FR1:",w_FR1)
plot.plot(x_points,poly.polynomial(x_points,w_FR1),label= '$CG$',color="purple")

w_FR2=CG.ConjungateGradientRegular(x,y,w_init)
print("w_FR2:",w_FR2)
plot.plot(x_points,poly.polynomial(x_points,w_FR2),label='$CG+Regular$',color="black")

plot.legend()
plot.show()
