import numpy as np
from math import log

def calc_entropy(dataset):
	"""
	计算熵，dataset为list
	"""
	rows = len(dataset)
	label_count = {}
	entropy = 0
	for i in range(rows):
		label = dataset[i][-1]
		label_count[label] = label_count.get(label,0) + 1
	for key,val in label_count.items():
		p = val / rows
		l = log(p,2) * (-1)
		entropy += p*l
	return entropy
	
def split_dataset_by_feature_value(dataset, feature_index, value):
	"""
	根据给定特征及其对应的特征值拆分数据集
	dataset：原始数据集
	feature_index：特征索引，也就是dataset的列索引
	value：feature_index对应的特征值
	"""
	new_dataset = []
	for row in dataset:
		if row[feature_index] == value:
			row_list = row[:feature_index]
			row_list.extend(row[feature_index+1:])
			new_dataset.append(row_list)
	return new_dataset
	
def choose_best_feature(dataset):
	"""
	根据信息增量，选择最佳分类特征
	"""
	base_entropy = calc_entropy(dataset)
	feature_num = len(dataset[0]) - 1 #特征个数
	max_info_gain = 0 #信息增量
	best_feature = -1 #最佳特征
	for feature_index in range(feature_num):
		feature_value_list = [row[feature_index] for row in dataset] #特征值列表
		feature_value_set = set(feature_value_list) #特征值集合，确保没有重复值
		new_entropy_sum = 0 #根据特征分类后的熵和
		info_gain = 0
		for value in feature_value_set:
			sub_dataset = split_dataset_by_feature_value(dataset, feature_index, value) #根据特定特征及其特征值得到子数据集
			prob = len(sub_dataset) / len(dataset)
			new_entropy = calc_entropy(sub_dataset)
			new_entropy_sum += prob * new_entropy
		info_gain = base_entropy - new_entropy_sum #计算信息增量
		if info_gain > max_info_gain:
			max_info_gain = info_gain
			best_feature = feature_index
	return best_feature
	
def max_count(classlist):
	"""
	返回出现次数最多的类别
	"""
	dic = {}
	for i in classlist:
		dic[classlist[i]] = dic.get(classlist[i],0)+1
	
	sorted_list = sorted(dic.items(), key=lambda x: x[1], reserver=True)
	print('sorted_list:',sorted_list)
	return sorted_list[0][0]
	
def create_policy_tree(dataset,lables):
	"""
	构建决策树
	"""
	classlist = [example[-1] for example in dataset]
	if classlist.count(classlist[0]) == len(classlist):  #类别完全相同停止继续分类
		return classlist[0]
	if len(dataset[0]) == 1:  #遍历完所有特征时返回出现次数最多的类别
		return max_count(classlist)
	best_feature = choose_best_feature(dataset)
	best_feature_lable = lables[best_feature]
	tree = {best_feature_lable:{}}
	del lables[best_feature]
	feature_value_list = [row[best_feature] for row in dataset] #特征值列表
	feature_value_set = set(feature_value_list) #特征值集合，确保没有重复值
	for value in feature_value_set:
		sub_dataset = split_dataset_by_feature_value(dataset, best_feature, value) #根据特定特征及其特征值得到子数据集
		tree[best_feature_lable][value] = create_policy_tree(sub_dataset,lables)
	return tree
	
if __name__ == '__main__':
	dataset = [[0,1,1],[0,1,1],[0,0,0],[1,1,0],[1,1,0]]
	lables = ['no surfacing','flippers']
	
	print("policy tree:",create_policy_tree(dataset,lables))