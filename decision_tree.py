#*--coding--=utf-8
header = ['色泽',	'根蒂',	'敲声',	'纹理',	'脐部',	'触感',	'好瓜']
'''dataset = [['-',   	'蜷缩',	'浊响',	'清晰',	'凹陷',	'硬滑',	'是'],
		   ['乌黑',	'蜷缩',	'沉闷',	'清晰',	'凹陷',	'-',	'是'],
		   ['乌黑',	'蜷缩',	'-',	'清晰',	'凹陷',	'硬滑',	'是'],
		   ['青绿',	'蜷缩',	'沉闷',	'清晰',	'凹陷',	'硬滑',	'是'],
		   ['-',   	'蜷缩',	'浊响',	'清晰',	'凹陷',	'硬滑',	'是'],
		   ['青绿',	'稍蜷',	'浊响',	'清晰',	'-',	'软粘',	'是'],
		   ['乌黑',	'稍蜷',	'浊响',	'稍糊',	'稍凹',	'软粘',	'是'],
		   ['乌黑',	'稍蜷',	'浊响',	'-',	'稍凹',	'硬滑',	'是'],
		   ['乌黑',	'-',	'沉闷',	'稍糊',	'稍凹',	'硬滑',	'否'],
		   ['青绿',	'硬挺',	'清脆',	'-',	'平坦',	'软粘',	'否'],
		   ['浅白',	'硬挺',	'清脆',	'模糊',	'平坦',	'-',	'否'],
		   ['浅白',	'蜷缩',	'-',	'模糊',	'平坦',	'软粘',	'否'],
		   ['-',   	'稍蜷',	'浊响',	'稍糊',	'凹陷',	'硬滑',	'否'],
		   ['浅白',	'稍蜷',	'沉闷',	'稍糊',	'凹陷',	'硬滑',	'否'],
		   ['乌黑',	'稍蜷',	'浊响',	'清晰',	'-',	'软粘',	'否'],
		   ['浅白',	'蜷缩',	'浊响',	'模糊',	'平坦',	'硬滑',	'否'],
		   ['青绿',	'-',	'沉闷',	'稍糊',	'稍凹',	'硬滑',	'否']]'''
		   
dataset = [['青绿', '蜷缩',	'浊响',	'清晰',	'凹陷',	'硬滑',	'是'],
		   ['乌黑',	'蜷缩',	'沉闷',	'清晰',	'凹陷',	'硬滑',	'是'],
		   ['乌黑',	'蜷缩',	'浊响',	'清晰',	'凹陷',	'硬滑',	'是'],
		   ['青绿',	'蜷缩',	'沉闷',	'清晰',	'凹陷',	'硬滑',	'是'],
		   ['浅白', '蜷缩',	'浊响',	'清晰',	'凹陷',	'硬滑',	'是'],
		   ['青绿',	'稍蜷',	'浊响',	'清晰',	'稍凹',	'软粘',	'是'],
		   ['乌黑',	'稍蜷',	'浊响',	'稍糊',	'稍凹',	'软粘',	'是'],
		   ['乌黑',	'稍蜷',	'浊响',	'清晰',	'稍凹',	'硬滑',	'是'],
		   ['乌黑',	'稍蜷',	'沉闷',	'稍糊',	'稍凹',	'硬滑',	'否'],
		   ['青绿',	'硬挺',	'清脆',	'清晰',	'平坦',	'软粘',	'否'],
		   ['浅白',	'硬挺',	'清脆',	'模糊',	'平坦',	'硬滑',	'否'],
		   ['浅白',	'蜷缩',	'浊响',	'模糊',	'平坦',	'软粘',	'否'],
		   ['青绿', '稍蜷',	'浊响',	'稍糊',	'凹陷',	'硬滑',	'否'],
		   ['浅白',	'稍蜷',	'沉闷',	'稍糊',	'凹陷',	'硬滑',	'否'],
		   ['乌黑',	'稍蜷',	'浊响',	'清晰',	'稍凹',	'软粘',	'否'],
		   ['浅白',	'蜷缩',	'浊响',	'模糊',	'平坦',	'硬滑',	'否'],
		   ['青绿',	'蜷缩',	'沉闷',	'稍糊',	'稍凹',	'硬滑',	'否']]
		
import math		
def entropy(data):
	rows = len(data)
	if rows == 0:
		return 0
	result = {}
	result_col = len(data[0])-1
	ent = 0.0
	log2 = lambda x:math.log(x) / math.log(2) 
	for row in data:
		flag = row[result_col]
		result[flag] = result.get(flag,0)+1
	for val in result.values():
		p = val / len(data)
		ent += -p*log2(p)
	return ent
	
def split_dataset(dataset,col,value):
	'''
	根据指定的列和列值对数据集进行拆分
	value可为int、float、str
	'''
	if isinstance(value,int) or isinstance(value,float):
		split_fun = lambda x:x >= value
	else:
		split_fun = lambda x: x == value
		
	result = []
	for row in dataset:
		if split_fun(row[col]):
			#result.append(row[0:col]+row[col+1:])
			result.append(row)
	return result

def get_col_lables(dataset,col):
	if len(dataset) == 0:
		return
	result = {}
	for row in dataset:
		result[row[col]] = result.get(row[col],0) + 1
	return result 
	
def get_best_future(dataset):
	if len(dataset) == 0:
		return
	current_ent = entropy(dataset)  #计算未进行数据拆分前的熵
	print('current_ent:',current_ent)
	max_gain = 0.0
	best_col = -1
	cols = len(dataset[0])
	for col in range(cols-1):
		ent = 0.0
		info_gain = 0.0
		gain = 0.0
		labels = get_col_lables(dataset,col)
		for label,value in labels.items():
			split_set = split_dataset(dataset,col,label)
			ent = entropy(split_set)
			info_gain += (ent * (value/len(dataset)))
		gain = current_ent - info_gain
		#print('col:',col,'gain:',gain)
		if gain > max_gain:
			max_gain = gain
			best_col = col
	print('best_col:',best_col)
	return best_col

def createtree(data_set,tree):
	'''初步以字典的形式输出树结构'''
	labellist = [row[-1] for row in data_set]
	if labellist.count(labellist[0]) == len(data_set): #类别完全相同则停止划分
		return labellist[0]
		
	if len(data_set[0]) == 1: #遍历完所有特征时返回出现次数最多的类别
		return data_set[0][0]
	
	col = get_best_future(data_set)
	future = header[col]
	print(future)
	label_dict = get_col_lables(data_set,col)
	
	tree[future] = {}
	for key,value in label_dict.items():
		tree[future][key] = {}
		tree[future][key]= createtree(split_dataset(data_set,col,key),tree[future][key])
		
	return tree
	
if __name__ == "__main__":
	col = get_best_future(dataset)
	tree = {}
	createtree(dataset,tree)
	print(tree)