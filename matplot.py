import numpy as np
import matplotlib.pyplot as plt

def calcAB(x,y):
	"""
	一元线性回归
	"""
	n = len(x)
	sumX, sumY, sumXY, sumXX = 0, 0, 0, 0
	for i in range(0, n):
		sumX += x[i]
		sumY += y[i]
		sumXX += x[i] * x[i]
		sumXY += x[i] * y[i]
	a = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX)
	b = (sumXX * sumY - sumX * sumXY) / (n * sumXX - sumX * sumX)
	return a, b
	
xi = [1,2,3,4,5,6,7,8,9,10]
yi = [10,11.5,12,13,14.5,15.5,16.8,17.3,18,18.7]
a,b=calcAB(xi,yi)
print("y = %10.5fx + %10.5f" %(a,b))

x = np.linspace(0,10)
y = a * x + b
plt.plot(x,y)
plt.scatter(xi,yi)
plt.show()



x = [1,2,3,4,5,6,7,8,9,10]
y = [10,11.5,12,13,14.5,15.5,16.8,17.3,18,18.7]
X = np.vstack([np.ones(len(x)),x]).T
Y = np.array(y).T
W=np.dot(np.matrix(np.dot(X.T,X))**-1,np.dot(X.T,Y))
yi=np.dot(X,W.T)#这里公式里是不需要转置的，但由于矩阵运算时W自动保存成一行多列的矩阵，所以多转置一下，配合原公式的计算。
print(X)
print(Y)
print(W)
print(yi)#拟合出的预测点
plt.plot(x,y,'o',label='data',markersize=10)
plt.plot(x,yi,'r',label='line')
plt.show()