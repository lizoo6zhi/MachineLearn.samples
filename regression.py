"""
	线性回归
	通过最小二乘法得到最优解
"""
import numpy as np
import matplotlib.pyplot as plt

def load_dataset(file):
	dataset = []
	label = []
	with open(file,'r') as pf:
		for line in pf:
			line_list = line.strip().split('\t')
			dataset.append([float(data) for data in line_list[0:2]])
			label.append(float(line_list[-1][0:-1]))
	return dataset,label
	
def stand_regres(dataset,label):
	"""
	标准线性回归函数
	dataset:list
	label:list
	"""
	X = np.mat(dataset)
	Y = np.mat(label).T

	temp = X.T * X
	if np.linalg.det(temp) == 0.0:  #计算行式的值
		print("X.T*X没有逆矩阵")
		return 
	else:
		temp = temp.I * X.T * Y
	return temp
	
def lwrw(test_point, xarray, yarray, k = 1.0):
	"""
	局部加权回归函数(locally weighted linear regression)
	dataset:list
	label:list
	"""
	mat_x = np.mat(xarray)
	mat_y = np.mat(yarray).T
	row,col = np.shape(mat_x)	
	weights = np.mat(np.eye(row))
	for i in range(row):
		diffmat = test_point - mat_x[i,:]
		weights[i,i] = np.exp((diffmat*diffmat.T)/(-2.0*k*k))
	temp = mat_x.T * weights * mat_x
	if np.linalg.det(temp) == 0.0:
		print("mat_x.T * weights * X没有逆矩阵")
		return 
	else:
		temp = temp.I * mat_x.T * weights * mat_y
	return test_point*temp

def data_normalize(dataset):
	"""
	数据标准化处理
	"""
	dataset_mat = np.mat(dataset)
	mean = np.mean(dataset_mat,0) #计算每行平均值
	print('mat mean:',mean)
	var = np.var(dataset,0) #计算每行标准差
	print('mat var:',var)
	return mean,var
	
def ridge_regres(xarray,yarray,lam=0.2):
	"""
	岭回归
	处理当特征比样本点还多，也酒水说输入数据的矩阵X不是满秩矩阵，非满秩矩阵在求逆时会出现问题
	w = (X.T*X + rI).I * X.T * Y
	"""
	xmat = np.mat(xarray)
	ymat = np.mat(yarray).T
	xtx = xmat.T * xmat
	denom = xtx + np.eye(np.shape(xmat)[1])*lam
	if np.linalg.det(denom) == 0.0:
		print('denom 没有逆矩阵')
	ws = denom.I * xmat.T * ymat
	return ws
	
def ridge_test(xarray,yarray,ax):
	xmat = np.mat(xarray)
	row,col = np.shape(xmat)
	ymat = np.mat(yarray)
	xmean,xvar = data_normalize(xarray) #对数据进行标准化处理
	np.seterr(divide='ignore',invalid='ignore')
	xmat = (xmat - xmean) / xvar
	num_test_pts = 30
	wmat = np.ones((num_test_pts,col))
	
	lam_list = []
	for i in range(num_test_pts):
		lam = np.exp(i-10)
		lam_list.append(i-10)
		ws = ridge_regres(xarray,yarray,lam)
		wmat[i,:] = ws.T
	
	#绘制不同lam情况下参数变化情况
	for j in range(np.shape(wmat)[1]):
		ax.plot(lam_list,wmat[:,j])
	
def plot_stand_regress(dataset,label,ws,ax):
	"""
	绘制数据集和拟合曲线
	"""
	dataset_matrix = np.mat(dataset)
	x = dataset_matrix[:,1]
	y = label
	cal_y = dataset * ws
	print(np.corrcoef(cal_y.T,np.mat(y)))
	ax.scatter(x.T.getA(),y,s=10,c='red')
	ax.plot(x,cal_y)
	
def plot_lwrw_regress(dataset,label,cal_y_list,ax):
	"""
	绘制数据集和拟合曲线
	"""
	dataset_matrix = np.mat(dataset)
	x = dataset_matrix[:,1]
	y = label
	ax.scatter(x.T.getA(),y,s=10,c='red')
	#对dataset进行排序
	sorted_index = x.argsort(0)
	sorted_x = dataset_matrix[sorted_index][:,0,:]
	ax.plot(sorted_x[:,1],cal_y_list[sorted_index])

		
if __name__ == "__main__":
	dataset,label = load_dataset('dataset/ex0.txt')
	fig = plt.figure()
	# 标准线性回归函数测试
	stand_ws = stand_regres(dataset,label)
	ax = fig.add_subplot(321)
	ax.text(0.05,4.2,'stand')
	plot_stand_regress(dataset,label,stand_ws,ax)
	
	#局部加权回归函数测试
	rows = len(dataset)
	x_y_map = np.zeros(rows)
	
	#当k为0.01时
	for row in range(rows):
		cal_y = lwrw(dataset[row],dataset,label,0.01)
		x_y_map[row] = cal_y
	ax = fig.add_subplot(323)
	ax.text(0.05,4.2,'k=0.01')
	plot_lwrw_regress(dataset,label,x_y_map,ax)
	
	#当k为0.003
	for row in range(rows):
		cal_y = lwrw(dataset[row],dataset,label,0.003)
		x_y_map[row] = cal_y
	ax = fig.add_subplot(325)
	ax.text(0.05,4.2,'k=0.003')
	plot_lwrw_regress(dataset,label,x_y_map,ax)
	
	#岭回归测试
	ws = ridge_regres(dataset,label)
	ax = fig.add_subplot(322)
	ax.text(0.05,4.2,'lam=0.2')
	plot_stand_regress(dataset,label,ws,ax)
	
	ax = fig.add_subplot(324)
	ax.text(10,2.5,'log(r)-->w')
	ridge_test(dataset,label,ax)
	
	plt.show()