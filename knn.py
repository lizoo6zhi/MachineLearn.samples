import numpy as np

def knn(labels, dataset, in_data,k):

	"""
	labels：list
	dataset：ndarray
	in_data：ndarray
	k：int
	"""
	row = dataset.shape[0]
	sub_set = dataset - np.tile(in_data,(row,1))
	sqrt_set = sub_set ** 2
	distance = sqrt_set.sum(axis=1) ** 0.5
	dic = {}
	#for index,value in enumerate(distance):  #算出来的距离与索引对应
		#dic[index] = value
	#sort_list = sorted(dic.items(), key=lambda x: x[1])
	
	# 使用内置函数argsort替换上面的sorted
	sortedIndex_list = distance.argsort()
	# 算出邻居中对应的label次数
	result = {}
	for i in range(k):
		key = sortedIndex_list[i]
		result[labels[key]] = result.get(labels[key],0)+1
		
	result_list=sorted(result.items(), key=lambda x: x[1],reverse=True)
	return result_list[0][0]

def file2matrix(filename):

	"""将文件数据转换为numpy.ndarray"""
	
	list_lines = []
	with open(filename,'r') as pf:
		lines = pf.readlines()
		for line in lines:
			list_lines.append([float(i) for i in line.strip().split('\t')])
			
	matrix = np.array(list_lines)
	data_set = matrix[:,0:-1]
	class_labels = matrix[:,-1:]
	class_labels = class_labels.reshape(1,1000).tolist()[0]
	return data_set,class_labels

def auto_norm(dataset):
	"""
	为了防止各特征值之间的数值差距太大而影响计算结果，
	对矩阵数据进行归一化处理，将每个特征值都转换为[0,1]
	"""
	min = dataset.min(0)
	max = dataset.max(0)
	sub = max - min
	norm_dataset = np.zeros(dataset.shape)
	row = dataset.shape[0]
	norm_dataset = dataset - np.tile(min,(row,1))
	norm_dataset = norm_dataset / np.tile(sub,(row,1))
	return norm_dataset,sub,min
	
def test_datingTestSet():
	data_set,date_labels = file2matrix('dataset\\datingTestSet2.txt')
	norm_dataset,sub,min = auto_norm(data_set)

	#根据数据集测试KNN算法的正确率
	rows = norm_dataset.shape[0]
	ok = 0
	for i in range(rows):
		result = knn(date_labels,norm_dataset,norm_dataset[i,:],3)
		if result == date_labels[i]:
			ok += 1
		else:
			print("测试结果为{}，正确结果为{}".format(result, date_labels[i]))
			print("测试数据{}失败".format(i+1))
	print("测试总记录数为{}，成功记录数为{}，KNN算法成功率为{}".format(rows,ok,ok/rows))
	
if __name__ == "__main__":
	test_datingTestSet()
		
	
	