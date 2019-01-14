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
			n = len(line_list)
			dataset.append([float(data) for data in line_list[0:-1]])
			label.append(float(line_list[-1]))
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
	数据标准化处data_normalize理
	"""
	dataset_mat = np.mat(dataset)
	mean = np.mean(dataset_mat,0) #计算每行平均值
	var = np.var(dataset,0) #计算每行标准差
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

def stage_wise(dataset,label,step,numiter=100):
	"""
	前向逐步回归
	"""
	xmat = np.mat(dataset)
	xmean,xvar = data_normalize(dataset) #平均值和标准差
	ymat = np.mat(label).T
	ymat = ymat - np.mean(ymat,0)
	xmat = (xmat - xmean) / xvar
	m,n = np.shape(xmat)
	return_mat = np.zeros((numiter,n))
	ws = np.zeros((n,1))
	ws_test = ws.copy()
	ws_max = ws.copy()
	
	for iter in range(numiter):  #迭代次数
		lowest_error = np.inf
		for j in range(n):  #对每个特征进行变量
			for sign in [-1,1]:
				ws_test = ws.copy()
				ws_test[j] += step*sign
				ytest = xmat * ws_test
				rssE = ((ymat.A-ytest.A)**2).sum()
				if rssE < lowest_error:
					lowest_error = rssE
					ws_max = ws_test
		ws = ws_max.copy()
		return_mat[iter,:] = ws.T
	return return_mat	

def stage_wise_test(ws_mat,iternum,ax):
	"""
	前向逐步回归算法中回归系数随迭代次数的变化情况
	"""	
	x = range(iternum)
	for i in range(np.shape(ws_mat)[1]):
		ax.plot(x,ws_mat[:,i])
					
if __name__ == "__main__":
	dataset,label = load_dataset('dataset/ex0.txt')
	fig = plt.figure()
	# 标准线性回归函数测试
	stand_ws = stand_regres(dataset,label)
	ax = fig.add_subplot(321)
	ax.text(0.05,4.2,'stand_regres')
	plot_stand_regress(dataset,label,stand_ws,ax)
	
	#局部加权回归函数测试
	rows = len(dataset)
	x_y_map = np.zeros(rows)
	
	#当k为0.01时
	for row in range(rows):
		cal_y = lwrw(dataset[row],dataset,label,0.01)
		x_y_map[row] = cal_y
	ax = fig.add_subplot(323)
	ax.text(0.05,4.2,'lwrw k=0.01')
	plot_lwrw_regress(dataset,label,x_y_map,ax)
	
	#当k为0.003
	for row in range(rows):
		cal_y = lwrw(dataset[row],dataset,label,0.003)
		x_y_map[row] = cal_y
	ax = fig.add_subplot(325)
	ax.text(0.05,4.2,'lwrw k=0.003')
	plot_lwrw_regress(dataset,label,x_y_map,ax)
	
	#岭回归测试
	ws = ridge_regres(dataset,label)
	ax = fig.add_subplot(322)
	ax.text(0.05,4.2,'ridge_regres lam=0.2')
	plot_stand_regress(dataset,label,ws,ax)
	
	dataset,label = load_dataset('dataset/abalone.txt')
	ax = fig.add_subplot(324)
	ax.text(10,2.5,'ridge_regres log(r)-->w')
	ridge_test(dataset,label,ax)
	
	#前向逐步回归
	ax = fig.add_subplot(326)
	
	ax.text(300,-0.4,'stage_wise 500')
	ws = stage_wise(dataset,label,0.01,500)
	print("前向逐步回归经过迭代后的最后回归系数：",ws[-1,:])
	stage_wise_test(ws,500,ax)
	
	#使用标准回归对"dataset/abalone.txt"数据集进行测试
	
	#对数据进行标准化处理
	xman,xvar = data_normalize(dataset)
	xmat = np.mat(dataset)
	xmat = (xmat - xman) / xvar
	ymat = np.mat(label)
	ax = stand_regres(xmat,ymat)
	print("标准回归后得到的回归系数：",ws[-1,:])
	plt.show()