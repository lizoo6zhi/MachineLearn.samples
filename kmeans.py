#*--coding=utf8--*
import matplotlib.pyplot as plt
import numpy as np
import random

def loadDataset(file):
	dataset = []

	with open(file,'r') as pf:
		for line in pf:
			li = line[:-1].split('\t')
			dataset_line = list(map(float,li))
			dataset.append(dataset_line)
	return dataset
	
def calDistance(vecA,vecB):
	distance = np.sqrt(np.sum(np.power(vecA-vecB,2)))
	return distance
	
def randKcent(datamat,k):
	m,n = np.shape(dataset)
	centerK = np.mat(np.ones((k,n)))
	for col in range(n):
		col_min = np.min(datamat[:,col])
		col_max = np.max(datamat[:,col])
		centerK[:,col] = col_min + (col_max-col_min)*np.random.rand(k,1)
	return centerK
	
def kMeans(datamat,k):
	"""
		根据K均值算法调整聚类中心点
	"""
	centerK = randKcent(datamat,k)
	m,n = np.shape(datamat)
	
	record_label_mat = np.zeros((m,2)) #记录每个数据行对应的分类索引和距离
	label_update = True
	while(label_update):
		label_update = False
		for i in range(m):
			min_distance = np.inf
			min_index = -1
			for j in range(k):
				distance = calDistance(datamat[i,:],centerK[j,:])
				if distance < min_distance:
					min_index = j
					min_distance = distance
			if record_label_mat[i,1] != min_distance:
				label_update = True #距离更新	
				record_label_mat[i,:] = min_index,min_distance
		#更新centerK	
		print(centerK)
		for i in range(k):
			ptsInClust = datamat[np.nonzero(record_label_mat[:,0] == i)[0]]
			centerK[i,:] = np.mean(ptsInClust,axis=0)
	return centerK,record_label_mat
	
def plotDataset(dataset,flagarr):
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	x = []
	y = []
	centerx = []
	centery = []
	for data in dataset:
		x.append(data[0])
		y.append(data[1])
	for flag in flagarr:
		centerx.append(flag[0])
		centery.append(flag[1])
	ax.scatter(x,y,s=30,c='black')
	ax.scatter(centerx,centery,s=30,c='red')
	plt.show()
	
if __name__ == "__main__":
	dataset = loadDataset('./dataset/kmeans/testSet.txt')
	centerK,record = kMeans(np.mat(dataset),4)
	plotDataset(dataset,centerK.A)
	