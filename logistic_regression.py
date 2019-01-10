import numpy as np
import math
import random
import matplotlib.pyplot as plt

def load_dataset():
	"""
	获得数据集
	"""
	dataset = []
	labels = []
	with open('dataset/testSet.txt','r') as pf:
		for line in pf:
			row_list = line.strip().split()
			dataset.append([1.0,float(row_list[0]),float(row_list[1])]) #三个特征x0,x1,x2
			labels.append(float(row_list[2]))
	return dataset,labels

def sigmoid(t):
	s = 1.0 / (1+np.exp(-t))
	return s
	
def grad_ascent(dataset,labels):
	"""
	回归梯度上升优化算法
	"""
	dataset_matrix = np.mat(dataset)
	label_matrix = np.mat(labels).transpose()
	row,col = np.shape(dataset_matrix)
	weight = np.ones((col,1))
	alpha = 0.01 #学习率
	max_cycles = 500
	for i in range(max_cycles):
		h = sigmoid(dataset_matrix * weight)
		error = label_matrix -h #真实值与预测值之间的误差
		temp = dataset_matrix.transpose() * error #交叉熵的偏导数
		weight = weight + alpha * temp  #更新权重
	return weight
	
def stoc_grad_ascent0(dataset,labels):
	"""
	随机梯度上升算法
	"""
	dataset_matrix = np.array(dataset)
	row,col = np.shape(dataset_matrix)
	weights = np.ones(col)
	alpha = 0.01
	for j in range(500):
		for i in range(row):
			h = sigmoid(sum(dataset_matrix[i]*weights))
			error = labels[i] - h
			weights = weights + alpha*error*dataset_matrix[i]
	return weights
	
def stoc_grad_ascent1(dataset,labels,numiter=150):
	"""
	优化随机梯度上升算法
	"""
	dataset_matrix = np.array(dataset)
	row,col = np.shape(dataset_matrix)
	weights = np.ones(col)
	for j in range(numiter):
		dataindex = range(row)
		for i in range(row):
			alpha = 4 / (1.0+j+i)+0.01
			randindex = int(random.uniform(0,len(dataindex)))
			h = sigmoid(sum(dataset_matrix[randindex]*weights))
			error = labels[randindex] - h
			weights = weights + alpha*error*dataset_matrix[randindex]
	return weights
	
def plot_bestfit(weights):
	dataset,labels = load_dataset()
	dataset_matrix = np.mat(dataset)
	rows,cols = np.shape(dataset_matrix)
	x0 = []
	y0 = []
	x1 = []
	y1 = []
	for row in range(rows):
		if labels[row] == 0:
			x0.append(dataset_matrix[row,1])
			y0.append(dataset_matrix[row,2])
		else:
			x1.append(dataset_matrix[row,1])
			y1.append(dataset_matrix[row,2])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(x0,y0,s=20,c='red',marker='s')
	ax.scatter(x1,y1,s=20,c='green')
	x = np.arange(-3.0,3.0,0.1)
	y = (-weights[0]-weights[1]*x)/weights[2]
	ax.plot(x,y)
	plt.xlabel('x')
	plt.ylabel('y')
	plt.show()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	以下代码为测试
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def stoc_grad_ascent0_heights_test(dataset,labels):
	dataset_matrix = np.array(dataset)
	row,col = np.shape(dataset_matrix)
	weights = np.ones(col)
	weights_iter_list = {}
	alpha = 0.01
	for j in range(500):
		weights_iter_list[j] = []
		for i in range(row):
			h = sigmoid(sum(dataset_matrix[i]*weights))
			error = labels[i] - h
			weights = weights + alpha*error*dataset_matrix[i]
			weights_iter_list[j].append(weights)
	return weights,weights_iter_list
	
def stoc_grad_ascent1_heights_test(dataset,labels,numiter=500):
	dataset_matrix = np.array(dataset)
	row,col = np.shape(dataset_matrix)
	weights = np.ones(col)
	weights_iter_list = {}
	for j in range(numiter):
		weights_iter_list[j] = []
		dataindex = range(row)
		for i in range(row):
			alpha = 4 / (1.0+j+i)+0.01
			randindex = int(random.uniform(0,len(dataindex)))
			h = sigmoid(sum(dataset_matrix[randindex]*weights))
			error = labels[randindex] - h
			weights = weights + alpha*error*dataset_matrix[randindex]
			weights_iter_list[j].append(weights)
	return weights,weights_iter_list
	
def plot_heights_change(weights_iter_list):
	x = []
	y0 = []
	y1 = []
	y2 = []
	for i,weights_list in weights_iter_list.items():
		for wieghts in weights_list:
			x.append(i)
			y0.append(wieghts[0])
			y1.append(wieghts[1])
			y2.append(wieghts[2])
	plt.subplot(311)
	plt.ylabel('w0')
	plt.plot(x,y0)
	plt.subplot(312)
	plt.ylabel('w1')
	plt.plot(x,y1)
	plt.subplot(313)
	plt.ylabel('w2')
	plt.plot(x,y2)
	plt.show()
	
if __name__ == "__main__":
	dataset,labels = load_dataset()
	A1 = grad_ascent(dataset,labels)
	h = sigmoid(np.mat(dataset)*A1)
	plot_bestfit(A1.getA())
	
	A2 = stoc_grad_ascent0(dataset,labels)
	plot_bestfit(A2)
	heights,weights_iter_list = stoc_grad_ascent0_heights_test(dataset,labels)
	plot_heights_change(weights_iter_list)
	A3 = stoc_grad_ascent1(dataset,labels)
	plot_bestfit(A3)
	heights,weights_iter_list = stoc_grad_ascent1_heights_test(dataset,labels)
	plot_heights_change(weights_iter_list)
	