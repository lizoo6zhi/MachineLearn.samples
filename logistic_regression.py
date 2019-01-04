import numpy as np
import math

"""
	sigmoid函数 s = 1 / (1+exp(-t))
"""
def load_dataset():
	dataset = []
	labels = []
	with open('testSet.txt','r') as pf:
		for line in pf:
			row_list = line.strip().split()
			dataset.append([1.0,float(row_list[0]),float(row_list[1])]) #三个特征x0,x1,x2
			labels.append(float(row_list[2]))
	return dataset,labels

def sigmoid(t):
	s = 1.0 / (1+np.exp(-t))
	return s
	
def grad_ascent(dataset,labels):
	dataset_matrix = np.mat(dataset)
	label_matrix = np.mat(labels).transpose()
	row,col = np.shape(dataset_matrix)
	weight = np.ones((col,1))
	alpha = 0.001 #学习率
	max_cycles = 500
	for i in range(max_cycles):
		h = sigmoid(dataset_matrix * weight)
		error = label_matrix -h #真实值与预测值之间的误差
		temp = dataset_matrix.transpose() * error
		weight = weight + alpha * temp  #更新权重
	return weight
	
def plot_bestfit(weights):
	print('weights:',weights)
	import matplotlib.pyplot as plt
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
	ax.scatter(x0,y0,s=30,c='red',marker='s')
	ax.scatter(x1,y1,s=30,c='green')
	x = np.arange(-3.0,3.0,0.1)
	y = (-weights[0]-weights[1]*x)/weights[2]
	ax.plot(x,y)
	plt.xlabel('x')
	plt.ylabel('y')
	plt.show()
	
if __name__ == "__main__":
	dataset,labels = load_dataset()
	A = grad_ascent(dataset,labels)
	print(A)
	h = sigmoid(np.mat(dataset)*A)
	print(h)
	plot_bestfit(A.getA())