#*--encoding=utf-8

"""
	封装系列机器学习算法
"""
标准线性回归(梯度下降和随机梯度下降)
W1 = W0 + step*dirtion
W1 = wo + step * X.T * (error)
标准线性回归(最小二乘法)
W = (X.T*X).I * X.T * Y
Tikhonov(吉洪洛夫)正则化
W = (X.T*X+a*np.eye(n)).I * X.T * Y  (a > 0)
lasso正则化